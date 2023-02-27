import numpy as np
import torch
from typing import Tuple

from LogisticCircuit.algo.BaseCircuit import BaseCircuit

from sklearn.linear_model._logistic import (_joblib_parallel_args)
from joblib import Parallel, delayed

from uncertainty_calculations import MonteCarloParams


def _monteCarloIteration(param: torch.Tensor, features: np.ndarray = None, prefix: str = '',
                         i: int = -1) -> torch.Tensor:
    """Single iteration for the parallel monte carlo prediction"""
    print(f"Evaluating {prefix} MC sample {i}                       ", end='\r')
    # despite the fact I am using a read only array for a read only operation in lgc.predict,
    # torch gives a warning that I cannot possibly silence, and since it happens on a thread it gets a lot of spam
    # since this is just a baseline it does not matter too much, so just silence the warning
    return torch.mm(torch.from_numpy(features.copy()), param.T)


def monteCarloPredictionParallel(trainingSampleMean: np.ndarray, lgc: BaseCircuit, params: MonteCarloParams,
                                 obsX: np.ndarray = None, prefix: str = '', jobs: int = -1
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is the baseline for monte carlo that disables batching, as batching is not possible in delta method due to
    a limitation in gradient calculation
    """
    # missing values
    # there may be a more vectorized way to write this, cannot think of one and its not a huge deal if its slower
    obsX = obsX.copy()
    for i in range(trainingSampleMean.shape[0]):
        # anywhere we see a -1 (missing), substitute in the training sample mean for that feature
        obsX[obsX[:, i] == -1, i] = trainingSampleMean[i]

    # compute features, only need to do once
    features = lgc.calculate_features(obsX)

    # collect samples for the first two moments, doing in a single loop saves effort setting the parameters
    delayedFunc = delayed(_monteCarloIteration)
    firstMoments = Parallel(n_jobs=jobs,
                            **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(param, features, prefix, i)
        for i, param in enumerate(params)
    )
    firstMoments = torch.concat(firstMoments, dim=1)

    # E[M1(phi)], var[M1[(phi)]
    return torch.mean(firstMoments, dim=1),\
        torch.var(firstMoments, dim=1)


def deltaMeanAndParameterVariance(trainingSampleMean: np.ndarray, lgc: BaseCircuit, params: torch.Tensor,
                                  obsX: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for delta method that ditches the small amount of batching we could do to ensure its comparable
    """
    if len(lgc.covariance) == 0:
        raise ValueError("Circuit must have covariance to compute a mean and variance")

    obsX = obsX.copy()
    for i in range(trainingSampleMean.shape[0]):
        # anywhere we see a -1 (missing), substitute in the training sample mean for that feature
        obsX[obsX[:, i] == -1, i] = trainingSampleMean[i]

    # E[M1[phi]],
    features = lgc.calculate_features(obsX)
    mean = torch.mm(torch.from_numpy(features), params.T)
    mean.backward(torch.ones((obsX.shape[0], lgc.num_classes)), inputs=params)
    grad = params.grad

    # var[M1[phi]], gradient vector is a row vector instead of a column vector
    # see above for note on stupid copying requirement
    # perhaps I should just use numpy to get the variance and make it torch after
    variance = torch.mm(grad, torch.mm(torch.from_numpy(lgc.covariance[0].copy()), grad.T))
    return mean.flatten(), variance
