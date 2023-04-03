"""
Utilities used as part of uncertainty functions. Kept separatate from utils.py and utils_missing.py to
make it easier to identify contribution.
"""
import gc
import numpy as np
from typing import Optional, Any, List, Tuple, Union

import torch
from joblib import delayed, Parallel
from sklearn.utils.fixes import _joblib_parallel_args
from torch import Tensor
from torch.distributions import Normal

from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode


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


def marginalizeGaussian(inputs: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces a marginal gaussian distribution over all the missing variables in the input vector
    """
    missingIndexes = inputs == -1
    return mean[missingIndexes], covariance[np.ix_(missingIndexes, missingIndexes)]


def conditionalGaussian(inputs: np.array, mean: np.ndarray, covariance: np.ndarray, returnCovariance: bool = True
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
    condVar = covariance[np.ix_(missingIndexes, missingIndexes)] \
        - np.matmul(np.matmul(corrMatrix, obsCovInv), corrMatrix.T)
    return condMean, condVar


def meanImputation(inputs: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Replaces all -1 in the dataset with the mean value from the given mean vector"""
    inputs = inputs.copy()
    for i in range(mean.shape[0]):
        # anywhere we see a -1 (missing), substitute in the training sample mean for that feature
        inputs[inputs[:, i] == -1, i] = mean[i]
    return inputs


def conditionalMeanImputation(inputs: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Replaces all -1 in the dataset with conditional mean given the input value"""
    inputs = inputs.copy()
    for i in range(inputs.shape[0]):
        image = inputs[i, :]
        missingIndexes = image == -1
        if missingIndexes.sum() != 0:
            # TODO: the following clamp is wrong and I should feel ashamed for writing it, but right now I need to test other stuff
            inputs[i, missingIndexes] = np.clip(
                conditionalGaussian(image, mean, covariance, returnCovariance=False),
                0, 1
            )
    return inputs
