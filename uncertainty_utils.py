"""
Utilities used as part of uncertainty functions. Kept separatate from utils.py and utils_missing.py to
make it easier to identify contribution.
"""
import gc
from typing import Optional, Any, List

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
