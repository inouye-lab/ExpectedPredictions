import torch
from typing import Tuple

from torch import Tensor

from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode
from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, \
    monteCarloPrediction, MonteCarloParams


def _gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    return torch.log(torch.div(1, torch.sqrt(var) * 2 * torch.pi)) + 0.5 / var * ((x - mean) ** 2)


def monteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, params: MonteCarloParams
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    size = (sampleCount,)
    error:               torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    inputLikelihood:     torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    parameterLikelihood: torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    totalLikelihood:     torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    inputVariances:      torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    parameterVariances:  torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    totalVariances:      torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    for i in range(sampleCount):
        # evaluate model
        feature = dataset.images[i, :].reshape(1, -1)
        mean, sampleParamVar, sampleInputVar = monteCarloPrediction(psdd, lgc, params, feature, prefix=f"var {i}")
        # add variances directly
        inputVariances[i]     = sampleInputVar
        parameterVariances[i] = sampleParamVar
        totalVariances[i]     = sampleInputVar + sampleParamVar
        # compute likelihoods
        y = dataset.labels[i]
        error[i] = torch.abs(mean - y)
        inputLikelihood[i]     = _gaussianLogLikelihood(y, mean, sampleInputVar)
        parameterLikelihood[i] = _gaussianLogLikelihood(y, mean, sampleParamVar)
        totalLikelihood[i]     = _gaussianLogLikelihood(y, mean, totalVariances[i])

    # return all six values
    return torch.sum(error), torch.mean(inputLikelihood), torch.mean(parameterLikelihood), torch.mean(totalLikelihood), \
        torch.mean(inputVariances), torch.mean(parameterVariances), torch.mean(totalVariances)


def deltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet
                               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param dataset: Dataset for computing the full value
    @return  Tuple of mean, parameter variance
    """
    count = dataset.labels.size()[0]
    size = (count,)
    error:               torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    inputLikelihood:     torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    parameterLikelihood: torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    totalLikelihood:     torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    inputVariances:      torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    parameterVariances:  torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    totalVariances:      torch.Tensor = torch.zeros(size=size, dtype=torch.float64)
    for i in range(count):
        # evaluate model
        cache = EVCache()
        feature = dataset.images[i, :].reshape(1, -1)
        mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
        sampleInputVar = deltaInputVariance(psdd, lgc, cache, feature)
        # add variances directly
        inputVariances[i]     = torch.abs(sampleInputVar)
        parameterVariances[i] = sampleParamVar
        totalVariances[i]     = inputVariances[i] + sampleParamVar
        # compute likelihoods
        y = dataset.labels[i]
        error[i] = torch.abs(mean - y)
        inputLikelihood[i]     = _gaussianLogLikelihood(y, mean, inputVariances[i])
        parameterLikelihood[i] = _gaussianLogLikelihood(y, mean, sampleParamVar)
        totalLikelihood[i]     = _gaussianLogLikelihood(y, mean, totalVariances[i])

    # return all six values
    return torch.sum(error), torch.mean(inputLikelihood), torch.mean(parameterLikelihood), torch.mean(totalLikelihood), \
        torch.mean(inputVariances), torch.mean(parameterVariances), torch.mean(totalVariances)
