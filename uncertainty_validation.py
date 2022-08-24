import copy
import numpy as np
import torch
from typing import Tuple

from torch import Tensor

from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode
from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, \
    monteCarloPrediction, MonteCarloParams, monteCarloPredictionParallel, exactDeltaTotalVariance

import gc
from sklearn.linear_model._logistic import (_joblib_parallel_args)
from joblib import Parallel, delayed


def _gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    return -0.5 * torch.log(torch.mul(2 * torch.pi, var))\
        - 0.5 / var * ((x - mean) ** 2)


def _baseGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int, function, *args
                               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Shared logic between any method that parallels over count
    @param psdd:      Probabilistic circuit root
    @param lgc:       Logistic or regression circuit
    @param dataset:   Dataset for computing the full value
    @param jobs:      Max number of parallel jobs to run, use -1 to use the max possible
    @param function:  Parallel function to run, handles the actual LL computation
    @param args:      Extra arguments to pass into the function
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    count = dataset.labels.size()[0]

    # prepare parallel
    delayedFunc = delayed(function)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(psdd, lgc, dataset.images[i, :], dataset.labels[i], i, *args)
        for i in range(count)
    )
    error, \
        inputLikelihood, parameterLikelihood, totalLikelihood, \
        inputVariances, parameterVariances, totalVariances = zip(*result)
    gc.collect()

    # return all six values
    return torch.sum(torch.tensor(error)), torch.mean(torch.tensor(inputLikelihood)),\
        torch.mean(torch.tensor(parameterLikelihood)), torch.mean(torch.tensor(totalLikelihood)), \
        torch.mean(torch.tensor(inputVariances)), torch.mean(torch.tensor(parameterVariances)),\
        torch.mean(torch.tensor(totalVariances))


def _monteCarloIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray,
                         y: torch.Tensor, i: int, params: MonteCarloParams) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the monte carlo method, used for parallel"""
    # evaluate model
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    mean, sampleParamVar, sampleInputVar = monteCarloPrediction(psdd, lgc, params, feature, prefix=f"var {i}")
    # add variances directly
    totalVariance = sampleInputVar + sampleParamVar
    # compute likelihoods
    error = torch.abs(mean - y)
    inputLikelihood = _gaussianLogLikelihood(y, mean, sampleInputVar)
    parameterLikelihood = _gaussianLogLikelihood(y, mean, sampleParamVar)
    totalLikelihood = _gaussianLogLikelihood(y, mean, totalVariance)
    return error, inputLikelihood, parameterLikelihood, totalLikelihood, sampleInputVar, sampleParamVar, totalVariance


def monteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, params: MonteCarloParams,
                                    jobs: int = -1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:      Probabilistic circuit root
    @param lgc:       Logistic or regression circuit
    @param dataset:   Dataset for computing the full value
    @param params:    Parameters to use for estimates
    @param jobs:       Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    return _baseGaussianLogLikelihood(psdd, lgc, dataset, jobs, _monteCarloIteration, params)


def fastMonteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, params: MonteCarloParams,
                                        jobs: int = -1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes likelihood and variances over the entire dataset,
    performing parallel over parameters and using torch to do parallel over inputs
    @param psdd:      Probabilistic circuit root
    @param lgc:       Logistic or regression circuit
    @param dataset:   Dataset for computing the full value
    @param params:    Parameters to use for estimates
    @param jobs:       Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # prepare parallel
    mean, parameterVariances, inputVariances = monteCarloPredictionParallel(psdd, lgc, params, dataset.images, jobs=jobs)

    totalVariances = parameterVariances + inputVariances
    error = torch.abs(mean - dataset.labels)
    inputLikelihood = _gaussianLogLikelihood(dataset.labels, mean, inputVariances)
    parameterLikelihood = _gaussianLogLikelihood(dataset.labels, mean, parameterVariances)
    totalLikelihood = _gaussianLogLikelihood(dataset.labels, mean, totalVariances)

    gc.collect()

    # return all six values
    return torch.sum(error), torch.mean(inputLikelihood),\
        torch.mean(parameterLikelihood), torch.mean(totalLikelihood), \
        torch.mean(inputVariances), torch.mean(parameterVariances),\
        torch.mean(totalVariances)


def _deltaGaussianIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the delta method, used for parallel"""

    print(f"Evaluating delta method {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    lgc.set_node_parameters(lgc.parameters.detach(), set_circuit=True, set_require_grad=True)


    mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
    # TODO: do I need a fresh cache?
    # TODO: why do I not pass in the mean?
    cache = EVCache()
    sampleInputVar = deltaInputVariance(psdd, lgc, cache, feature)
    # add variances directly
    inputVariance = torch.clamp(sampleInputVar, min=0)  # was abs before
    # TODO validated why it goes negative
    totalVariance = inputVariance + sampleParamVar
    # compute likelihoods
    error = torch.abs(mean - y)
    inputLikelihood = _gaussianLogLikelihood(y, mean, inputVariance)
    parameterLikelihood = _gaussianLogLikelihood(y, mean, sampleParamVar)
    totalLikelihood = _gaussianLogLikelihood(y, mean, totalVariance)
    return error, inputLikelihood, parameterLikelihood, totalLikelihood, inputVariance, sampleParamVar, totalVariance


def deltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1
                               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @param jobs:    Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    return _baseGaussianLogLikelihood(psdd, lgc, dataset, jobs, _deltaGaussianIteration)


def _exactDeltaGaussianIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                                 ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the exact delta method, used for parallel"""

    print(f"Evaluating delta method {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    lgc.set_node_parameters(lgc.parameters.detach(), set_circuit=True, set_require_grad=True)

    mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
    # TODO: do I need a fresh cache?
    # TODO: why do I not pass in the mean?
    cache = EVCache()
    totalVariance = exactDeltaTotalVariance(psdd, lgc, cache, feature)
    # add variances directly
    # inputVariance = torch.clamp(sampleInputVar, min=0)  # was abs before
    # TODO validated why it goes negative
    inputVariance = totalVariance - sampleParamVar
    # compute likelihoods
    error = torch.abs(mean - y)
    inputLikelihood = _gaussianLogLikelihood(y, mean, inputVariance)
    parameterLikelihood = _gaussianLogLikelihood(y, mean, sampleParamVar)
    totalLikelihood = _gaussianLogLikelihood(y, mean, totalVariance)
    return error, inputLikelihood, parameterLikelihood, totalLikelihood, inputVariance, sampleParamVar, totalVariance


def exactDeltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @param jobs:    Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    return _baseGaussianLogLikelihood(psdd, lgc, dataset, jobs, _exactDeltaGaussianIteration)
