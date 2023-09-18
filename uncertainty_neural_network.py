import gc

import numpy as np
import torch
from numpy.random import RandomState

from LogisticCircuit.algo.NeuralNetworkBaseline import NeuralNetworkRegressor
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode
from uncertainty_utils import conditionalGaussian, augmentMonteCarloSamplesGaussian, augmentMonteCarloSamplesPSDD, \
    orElse
from uncertainty_validation import _summarize, SummaryFunction, SummaryType


def basicNNImputation(imputationFunction: callable, nn: NeuralNetworkRegressor, dataset: DataSet,
                      summaryFunction: SummaryFunction = _summarize, residualUncertainty: float = 0,
                      inputUncertainty: torch.Tensor = None) -> SummaryType:
    """
    Handles missing values by replacing them with something else
    @param imputationFunction:  Logic to inject missing values
    @param nn:                  Neural network to evaluate
    @param dataset:             Dataset for computing the full value
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @param inputUncertainty:    Input uncertainty from residual method
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """

    obsX = imputationFunction(dataset.images)
    mean = nn(torch.from_numpy(obsX))
    variance = torch.zeros(size=mean.shape, dtype=torch.float)
    if inputUncertainty is None:
        inputUncertainty = variance

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputUncertainty, variance, inputUncertainty + residualUncertainty
    )


def monteCarloGaussianNNInputOnlyLogLikelihood(nn: NeuralNetworkRegressor,
                                               inputMean: np.ndarray, inputCovariance: np.ndarray, inputSamples: int,
                                               inputCovarianceInv: np.ndarray = None,
                                               inputReducer: callable = conditionalGaussian,
                                               randState: RandomState = None, enforceBoolean: bool = True,
                                               dataset: DataSet = None, summaryFunction: SummaryFunction = _summarize,
                                               residualUncertainty: float = 0
                                               ) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset using the gaussian method for missing values/uncertainty
    performing parallel over parameters and using torch to do parallel over test samples and input samples
    @param nn:                  Neural network to evaluate
    @param dataset:             Dataset for computing the full value
    @param inputMean:           Mean vector of the input distribution
    @param inputCovariance:     Covariance matrix of the input distribution
    @param inputCovarianceInv:  Inverted covariance matrix of the input distribution, if None will compute as needed
    @param inputSamples:        Number of samples to take from the input distribution
    @param inputReducer:        Function to reduce the random variables, typically marginal or conditional
    @param enforceBoolean:      If true, enforces gaussian samples are boolean values.
                                Requires special training of the neural network to not enforce
    @param randState:           Random state for sampling inputs
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    obsXAugmented = augmentMonteCarloSamplesGaussian(
        inputMean, inputCovariance, inputSamples, inputCovarianceInv, inputReducer, dataset.images,
        randState=randState, enforceBoolean=enforceBoolean
    )

    # TODO: following is duplicate, can do something with an augmenter function
    testSamples = dataset.images.shape[0]
    features = dataset.images.shape[1]
    obsXInput = obsXAugmented.reshape(testSamples * inputSamples, features)

    predictions = nn(torch.from_numpy(obsXInput).float()).reshape(testSamples, inputSamples)

    # average over dimension 1 to get mean and covariances to return
    mean = torch.mean(predictions, dim=1).squeeze()
    inputVariances = torch.var(predictions, dim=1).squeeze()
    totalVariances = inputVariances + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputVariances, torch.zeros(size=inputVariances.shape, dtype=torch.float), totalVariances
    )


def monteCarloPSDDNNInputOnlyLogLikelihood(psdd: PSddNode, nn: NeuralNetworkRegressor, inputSamples: int,
                                           randState: RandomState = None, dataset: DataSet = None,
                                           summaryFunction: SummaryFunction = None, residualUncertainty: float = 0
                                           ) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset using the monte carlo sampling on a PSDD for missing
    values/uncertainty performing parallel over parameters and using torch to do parallel over input and test samples
    @param psdd:                Probabilistic circuit root
    @param nn:                  Neural network to evaluate
    @param dataset:             Dataset for computing the full value
    @param inputSamples:        Number of samples to take from the input distribution
    @param randState:           Random state for sampling inputs
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    obsXAugmented = augmentMonteCarloSamplesPSDD(psdd, inputSamples, dataset.images, randState=randState)

    testSamples = dataset.images.shape[0]
    features = dataset.images.shape[1]
    obsXInput = obsXAugmented.reshape(testSamples * inputSamples, features)

    predictions = nn(torch.from_numpy(obsXInput).float()).reshape(testSamples, inputSamples)

    # average over dimension 1 to get mean and covariances to return
    mean = torch.mean(predictions, dim=1)
    inputVariances = torch.var(predictions, dim=1)
    totalVariances = inputVariances + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputVariances, torch.zeros(size=inputVariances.shape, dtype=torch.float), totalVariances
    )
