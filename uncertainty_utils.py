"""
Utilities used as part of uncertainty functions. Kept separatate from utils.py and utils_missing.py to
make it easier to identify contribution.
"""
import gc
import numpy as np
from typing import Optional, Any, List, Tuple, Union

import pandas
import torch
from joblib import delayed, Parallel
from numpy.random import RandomState
from sklearn.utils.fixes import _joblib_parallel_args
from statsmodels.imputation import mice
from torch import Tensor
from torch.distributions import Normal

from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode, InstMap


def orElse(value: Optional[Any], fallback: Any):
    """
    Returns value if not none, or fallback if value is none
    """
    if value is None:
        return fallback
    return value


def gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    clampVar = var.clamp(min=1e-10)
    return -0.5 * torch.log(torch.mul(2 * torch.pi, clampVar))\
        - 0.5 / clampVar * ((x - mean) ** 2)


def gaussianPValue(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Computes the p-values of the true value for all samples in the mean and variance vectors"""
    normal = Normal(mean, torch.sqrt(var.clamp(min=1e-10)))
    return 2 * normal.cdf(mean - torch.abs(mean - x))


def confidenceError(pValues: torch.Tensor) -> torch.Tensor:
    """Computes the error between the ideal confidence values and the true one"""
    sortedValues, _ = torch.sort(1 - pValues)
    # the linspace represents the ideal value at each index, while sorted values are the actual value we got
    return torch.mean(torch.abs(sortedValues - torch.linspace(0, 1, pValues.shape[0])))


def parallelOverSamples(dataset: DataSet, jobs: int, function, *args) -> List[Tensor]:
    """
    Shared logic between any method that parallels over count
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
        delayedFunc(dataset.images[i, :], dataset.labels[i], i, *args)
        for i in range(count)
    )
    resultTensors: List[Optional[torch.Tensor]] = [None]*len(result[0])
    for i, result in enumerate(zip(*result)):
        resultTensors[i] = torch.tensor(result)
    gc.collect()
    return resultTensors


# noinspection PyUnusedLocal
# Extra parameter used for parity with conditional gaussian
def marginalizeGaussian(inputs: np.ndarray, mean: np.ndarray, covariance: np.ndarray, covarianceInv: np.ndarray = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces a marginal gaussian distribution over all the missing variables in the input vector
    """
    missingIndexes = inputs == -1
    return mean[missingIndexes], covariance[np.ix_(missingIndexes, missingIndexes)]


def conditionalGaussian(inputs: np.array, mean: np.ndarray, covariance: np.ndarray, covarianceInv: np.ndarray = None,
                        returnCovariance: bool = True
                        ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Produces a conditional gaussian distribution over all the missing variables in the input vector given observed
    """
    # If no observed inputs, nothing to do
    observedIndexes = inputs != -1
    if observedIndexes.sum() == 0:
        if returnCovariance:
            return mean, covariance
        return mean
    # If no missing indexes, nothing we can do
    missingIndexes = inputs == -1
    if missingIndexes.sum() == 0:
        if returnCovariance:
            return None, None
        return None

    # Partition of covariance containing covariances between missing indexes and observed indexes
    corrMatrix = covariance[np.ix_(missingIndexes, observedIndexes)]
    # Inverted partition of covariance containing just observed indexes
    obsCovInv = np.linalg.pinv(covariance[np.ix_(observedIndexes, observedIndexes)])
    # Final computed conditional mean
    condMean = mean[missingIndexes] + np.matmul(
        np.matmul(corrMatrix, obsCovInv),
        inputs[observedIndexes] - mean[observedIndexes]
    )
    # Quick exit if we do not care about the conditional covariance
    if not returnCovariance:
        return condMean

    # Final computed conditional variance
    # The following formula is more efficient (no need to compute an additional inverse)
    # but leads to a crash on some datasets as it produces a matrix that is not positive semi-definiate
    # condVar = covariance[np.ix_(missingIndexes, missingIndexes)] \
    #     - np.matmul(np.matmul(corrMatrix, obsCovInv), corrMatrix.T)

    # this formula is mathematically equivalent, but is less efficient
    # however, we should be guaranteed a valid covariance matrix at the end
    if covarianceInv is None:
        covarianceInv = np.linalg.pinv(covariance)
    condVar = np.linalg.pinv(covarianceInv[np.ix_(missingIndexes, missingIndexes)])
    return condMean, condVar


def meanImputation(inputs: np.ndarray, mean: np.ndarray, enforceBoolean: bool = True) -> np.ndarray:
    """Replaces all -1 in the dataset with the mean value from the given mean vector"""
    inputs = inputs.copy()
    meanVals = (mean > 0.5).astype(float) if enforceBoolean else mean
    for i in range(mean.shape[0]):
        # anywhere we see a -1 (missing), substitute in the training sample mean for that feature
        inputs[inputs[:, i] == -1, i] = meanVals[i]
    return inputs


def conditionalMeanImputation(inputs: np.ndarray, mean: np.ndarray, covariance: np.ndarray,
                              enforceBoolean: bool = True) -> np.ndarray:
    """Replaces all -1 in the dataset with conditional mean given the input value"""
    inputs = inputs.copy()
    for i in range(inputs.shape[0]):
        image = inputs[i, :]
        missingIndexes = image == -1
        if missingIndexes.sum() != 0:
            # TODO: the following clamp is wrong and I should feel ashamed for writing it, but right now I need to test other stuff
            condMeans = conditionalGaussian(image, mean, covariance, returnCovariance=False)
            if enforceBoolean:
                inputs[i, missingIndexes] = (condMeans > 0.5).astype(float)
            else:
                inputs[i, missingIndexes] = np.clip(condMeans, 0, 1)
    return inputs


def psddMpeImputation(inputs: np.ndarray, psdd: PSddNode) -> np.ndarray:
    """Replaces all -1 in the dataset with the MPE value from the PSDD"""
    inputs = inputs.copy()

    for sample in range(inputs.shape[0]):
        image = inputs[sample, :]
        missingIndexes = image == -1
        if missingIndexes.sum() != 0:
            instMap = InstMap.from_list(image)
            _, psddMpe = psdd.mpe(instMap)
            for i, isMissing in enumerate(missingIndexes):
                if isMissing:
                    inputs[sample, i] = psddMpe[i + 1]
                else:
                    assert inputs[sample, i] == psddMpe[i + 1]
    return inputs


def miceImputation(inputs: np.ndarray, additionalData: np.ndarray = None,
                   iterations: int = 1, enforceBoolean: bool = True) -> np.ndarray:
    """Replaces all -1 in the dataset with the MICE value"""

    # mice wants NAN for inputs
    inputs = inputs.copy()
    inputs[inputs == -1] = np.nan
    inputShape = inputs.shape

    # this might be wrong, but intuitively I feel MICE has a better chance if you give it info that has non-missing data
    # this is required to perform the residual method notably
    if additionalData is not None:
        inputs = np.concatenate((additionalData, inputs), axis=0)

    # just run MICE from the library, requires creating a data frame with each feature as a column
    mouse = mice.MICEData(pandas.DataFrame(inputs, columns=[
        "feature_" + str(i) for i in range(inputs.shape[1])
    ]))
    mouse.update_all(iterations)
    result = mouse.next_sample().values

    # if we augmented with additional data, remove it from the mice results
    if additionalData is not None:
        result = result[additionalData.shape[0]:, :]
    assert result.shape == inputShape, "Expected result shape " + str(result.shape) \
                                       + " to be the same as input shape " + str(inputShape)

    # MICE may give values outside the 0 to 1 range, ideally we should convert to booleans, but just clip if requested
    if enforceBoolean:
        return (result > 0.5).astype(np.float32)
    else:
        return np.clip(result, 0, 1).astype(np.float32)


def augmentMonteCarloSamplesGaussian(inputMean: np.ndarray, inputCovariance: np.ndarray, inputSamples: int,
                                     inputCovarianceInv: np.ndarray, inputReducer: callable, obsX: np.ndarray = None,
                                     seed: int = 1337, randState: RandomState = None,
                                     enforceBoolean: bool = True) -> np.ndarray:
    """
    Augments the observed X values with monte carlo samples from a gaussian input distribution
    """
    # the goal here is to process all test samples and all input samples in one large batch of size test * input
    # start by constructing a 3D matrix of test sample * input sample * feature
    testSamples = obsX.shape[0]
    features = obsX.shape[1]
    obsXAugmented = np.zeros((testSamples, inputSamples, features))

    # random state can be passed in or a seed is passed for threads
    if randState is None:
        randState = RandomState(seed)

    # form matrix from samples
    for sample in range(testSamples):
        image = obsX[sample, :]
        missingIndexes = image == -1

        obsXAugmented[sample, :, :] = np.repeat(image.reshape(1, 1, -1), repeats=inputSamples, axis=1)
        # If no missing indexes, no need to handle samples
        if missingIndexes.sum() != 0:
            # Need to sample the the distribution the requested number of times, then replace all -1 values
            missingMean, missingCov = inputReducer(image, inputMean, inputCovariance, inputCovarianceInv)
            missingSamples = randState.multivariate_normal(missingMean, missingCov, size=inputSamples)
            # TODO: the following clamp is wrong and I should feel ashamed for writing it, but right now I need to test other stuff
            if enforceBoolean:
                obsXAugmented[sample, :, missingIndexes] = (missingSamples.T > 0.5).astype(float)
            else:
                obsXAugmented[sample, :, missingIndexes] = np.clip(missingSamples.T, 0, 1)

        # we should have filled in all -1 values in the final array
        assert (obsXAugmented[sample, :, :] == -1).sum() == 0

    return obsXAugmented


def augmentMonteCarloSamplesPSDD(psdd: PSddNode, inputSamples: int, obsX: np.ndarray = None,
                                 seed: int = 1337, randState: RandomState = None) -> np.ndarray:
    """
    Augments the observed X values with monte carlo samples from a PSDD
    """
    testSamples = obsX.shape[0]
    features = obsX.shape[1]
    obsXAugmented = np.zeros((testSamples, inputSamples, features))

    # random state can be passed in or a seed is passed for threads
    if randState is None:
        randState = RandomState(seed)

    for sample in range(testSamples):
        image = obsX[sample, :]
        missingIndexes = image == -1

        obsXAugmented[sample, :, :] = np.repeat(image.reshape(1, 1, -1), repeats=inputSamples, axis=1)
        # If no missing indexes, no need to handle samples
        if missingIndexes.sum() != 0:
            # Need to sample the the distribution the requested number of times, then replace all -1 values
            instMap = InstMap.from_list(image)
            psdd.value(instMap, clear_data = False)
            for inputSample in range(inputSamples):
                psddSample = psdd.simulate_with_evidence(instMap.copy(), seed=randState.randint(0, 2**32 - 1))
                for i, isMissing in enumerate(missingIndexes):
                    if isMissing:
                        obsXAugmented[sample, inputSample, i] = psddSample[i + 1]
                    else:
                        assert obsXAugmented[sample, inputSample, i] == psddSample[i + 1]
            psdd.clear_bits()

        # we should have filled in all -1 values in the final array
        assert (obsXAugmented[sample, :, :] == -1).sum() == 0

    return obsXAugmented
