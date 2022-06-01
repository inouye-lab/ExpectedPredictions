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
    monteCarloPrediction, MonteCarloParams

import gc
from sklearn.linear_model._logistic import (_joblib_parallel_args)
from joblib import Parallel, delayed


def _gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    return -0.5 * torch.log(torch.mul(2 * torch.pi, var))\
        + 0.5 / var * ((x - mean) ** 2)


def _monteCarloIteration(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams, feature: np.ndarray,
                         y: torch.Tensor, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    @pa
    @return  Tuple of mean, parameter variance
    """
    sampleCount = dataset.labels.size()[0]

    # prepare parallel
    delayedFunc = delayed(_monteCarloIteration)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(psdd, lgc, params, dataset.images[i, :], dataset.labels[i], i)
        for i in range(sampleCount)
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


def _deltaGaussianIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluates a single iteration of the delta method, used for parallel"""

    print(f"Evaluating delta method {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    lgc._parameters = lgc._parameters.clone().detach()
    lgc._parameters.requires_grad = True
    lgc.set_node_parameters(lgc._parameters)

    mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
    sampleInputVar = deltaInputVariance(psdd, lgc, cache, feature)
    # add variances directly
    inputVariance = torch.abs(sampleInputVar)
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
    @return  Tuple of mean, parameter variance
    """
    count = dataset.labels.size()[0]

    # prepare parallel
    delayedFunc = delayed(_deltaGaussianIteration)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(psdd, lgc, dataset.images[i, :], dataset.labels[i], i)
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
