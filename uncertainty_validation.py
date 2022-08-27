import copy
import numpy as np
import torch
from typing import Tuple, List, Optional

from torch import Tensor

import uncertainty_baseline
from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.algo.RegressionCircuit import RegressionCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode
from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, \
    monteCarloPrediction, MonteCarloParams, monteCarloPredictionParallel, exactDeltaTotalVariance

import gc
from sklearn.linear_model._logistic import (_joblib_parallel_args)
from joblib import Parallel, delayed


SummaryType = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def _gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    clampVar = var.clamp(min=1e-10)
    return -0.5 * torch.log(torch.mul(2 * torch.pi, clampVar))\
        - 0.5 / clampVar * ((x - mean) ** 2)


def _baseParallelOverSamples(psdd: Optional[PSddNode], lgc: BaseCircuit, dataset: DataSet, jobs: int, function, *args
                             ) -> List[Tensor]:
    """
    Shared logic between any method that parallels over count
    @param psdd:      Probabilistic circuit root
    @param lgc:       Logistic or regression circuit
    @param dataset:   Dataset for computing the full value
    @param jobs:      Max number of parallel jobs to run, use -1 to use the max possible
    @param function:  Parallel function to run, handles the actual LL computation
    @param args:      Extra arguments to pass into the function
    @return  Tuple of function returns
    """
    count = dataset.labels.size()[0]

    # prepare parallel
    delayedFunc = delayed(function)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(psdd, lgc, dataset.images[i, :], dataset.labels[i], i, *args)
        for i in range(count)
    )
    resultTensors: List[Optional[torch.Tensor]] = [None]*len(result[0])
    for i, result in enumerate(zip(*result)):
        resultTensors[i] = torch.tensor(result)
    gc.collect()
    return resultTensors


def _summarize(dataset: DataSet, mean: torch.Tensor,
               inputVariances: torch.Tensor, parameterVariances: torch.Tensor, totalVariances: torch.Tensor
               ) -> SummaryType:
    """
    Helper to summarize the results of an experiment

    @param dataset             The dataset used for the experiment
    @param mean                Tensor of averages with a value per sample in the dataset
    @param inputVariances      Tensor of input variances with a value per sample in the dataset
    @param parameterVariances  Tensor of parameter variances with a value per sample in the dataset
    @param totalVariances      Tensor of total variances with a value per sample in the dataset
    @return  Total error, average loglikelihood for each variance, average variance
    """
    error = torch.abs(mean - dataset.labels)
    inputLikelihood = _gaussianLogLikelihood(dataset.labels, mean, inputVariances)
    parameterLikelihood = _gaussianLogLikelihood(dataset.labels, mean, parameterVariances)
    totalLikelihood = _gaussianLogLikelihood(dataset.labels, mean, totalVariances)

    return torch.sum(error), torch.mean(inputLikelihood),\
        torch.mean(parameterLikelihood), torch.mean(totalLikelihood), \
        torch.mean(inputVariances), torch.mean(parameterVariances),\
        torch.mean(totalVariances)


def _monteCarloIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray,
                         y: torch.Tensor, i: int, params: MonteCarloParams) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the monte carlo method, used for parallel"""
    # evaluate model
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    mean, sampleParamVar, sampleInputVar = monteCarloPrediction(psdd, lgc, params, feature, prefix=f"var {i}")
    # add variances directly
    return mean, sampleInputVar, sampleParamVar


def monteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, params: MonteCarloParams,
                                    jobs: int = -1) -> SummaryType:
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
    mean, inputVariances, parameterVariances = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _monteCarloIteration, params)
    totalVariances = parameterVariances + inputVariances

    gc.collect()
    return _summarize(dataset, mean, inputVariances, parameterVariances, totalVariances)


def fastMonteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, params: MonteCarloParams,
                                        jobs: int = -1) -> SummaryType:
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

    gc.collect()
    return _summarize(dataset, mean, inputVariances, parameterVariances, totalVariances)


def _deltaIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                            ) -> Tuple[Tensor, Tensor]:
    """Evaluates a single iteration of the delta method, used for parallel"""
    print(f"Evaluating delta method for sample {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    lgc.set_node_parameters(lgc.parameters.detach(), set_circuit=True, set_require_grad=True)

    mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
    return mean, sampleParamVar


def deltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1
                               ) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @param jobs:    Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _deltaIteration)

    # can easily batch input variance as we are not taking gradients there
    inputVariance, _ = deltaInputVariance(psdd, lgc, EVCache(), dataset.images)
    # add variances directly
    inputVariance = torch.clamp(inputVariance, min=0)
    totalVariance = inputVariance + paramVariance

    gc.collect()
    return _summarize(dataset, mean, inputVariance, paramVariance, totalVariance)


def _exactDeltaIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                                 ) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the exact delta method, used for parallel"""

    print(f"Evaluating delta method for sample {i}", end='\r')
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
    return mean, sampleParamVar, totalVariance


def exactDeltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @param jobs:    Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance, totalVariance = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _exactDeltaIteration)
    inputVariance = totalVariance - paramVariance

    gc.collect()
    return _summarize(dataset, mean, inputVariance, paramVariance, totalVariance)


def monteCarloParamLogLikelihood(trainingSampleMean: np.ndarray, lgc: BaseCircuit, dataset: DataSet,
                                 params: MonteCarloParams, jobs = -1) -> SummaryType:
    """
    Baseline function to test the parameter variance using the monte carlo method without considering input variance.
    Missing values are handled through a training sample mean.

    @param trainingSampleMean: Average value of each feature in the training data
    @param lgc:                Circuit instance
    @param dataset:            Dataset for computing the full value
    @param params:             Parameters to use for estimates
    @param jobs:    Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # prepare parallel
    mean, parameterVariances = uncertainty_baseline.monteCarloPredictionParallel(
        trainingSampleMean, lgc, params, dataset.images, prefix='Baseline Param', jobs=jobs)

    gc.collect()
    return _summarize(dataset, mean, torch.tensor(0, dtype=torch.float), parameterVariances, parameterVariances)


def _deltaParamIteration(ignored: None, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int,
                         trainingSampleMean: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """Evaluates a single iteration of the delta method, used for parallel"""

    print(f"Evaluating delta method for sample {i}", end='\r')
    feature = feature.reshape(1, -1)
    # clone the parameters to prevent conflicting the gradients with the other threads
    params = lgc.parameters.detach().clone()
    params.requires_grad = True
    mean, sampleParamVar = uncertainty_baseline.deltaMeanAndParameterVariance(trainingSampleMean, lgc, params, feature)
    return mean, sampleParamVar


def deltaParamLogLikelihood(trainingSampleMean: np.ndarray, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1):
    """
    Baseline function to test the parameter variance using the delta method without considering input variance.
    Missing values are handled through a training sample mean.

    @param trainingSampleMean: Average value of each feature in the training data
    @param lgc:                Circuit instance
    @param dataset:            Dataset for computing the full value
    @param jobs:               Max number of parallel jobs to run, use -1 to use the max possible
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance = _baseParallelOverSamples(None, lgc, dataset, jobs, _deltaParamIteration, trainingSampleMean)

    gc.collect()
    return _summarize(dataset, mean, torch.tensor(0, dtype=torch.float), paramVariance, paramVariance)


def inputLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet) -> SummaryType:
    """
    Computes likelihood and input variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # delta method for input variance is already essentially the same as only considering input variance
    # difference is just the training method (and the fact delta normally gets param variance)
    inputVariance, mean = deltaInputVariance(psdd, lgc, EVCache(), dataset.images)
    inputVariance = torch.clamp(inputVariance, min=0)

    gc.collect()
    return _summarize(dataset, mean, inputVariance, torch.tensor(0, dtype=torch.float), inputVariance)
