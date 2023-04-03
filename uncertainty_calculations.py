import copy

import logging
import numpy as np
import torch
from typing import Tuple, List, Optional

from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from circuit_expect import Expectation, moment
from numpy.random import RandomState
from pypsdd import PSddNode

# noinspection PyProtectedMember
from sklearn.linear_model._logistic import (_joblib_parallel_args)
from joblib import Parallel, delayed

from uncertainty_utils import conditionalGaussian

MonteCarloParams = List[torch.Tensor]
"""List of params for monte carlo, taking advantage of the fact we can reuse a cache for a given input vector"""


def sampleMonteCarloParameters(lgc: BaseCircuit, count: int, randState: RandomState = None) -> MonteCarloParams:
    """
    Computes the variance over the parameters for the circuit using monte carlo
    @param lgc:       Logistic or regression circuit
                      takes advantage of the fact we can reuse a cache for a given value
    @param count:     Number of samples to take
    @param randState: Seeded generator for reproducibility, if unset creates
    @return  List of parameters for the samples
    """
    if len(lgc.covariance) == 0:
        raise ValueError("Circuit must have covariance to compute a mean and variance")
    if randState is None:
        randState = RandomState(1337)

    # generate all the samples, then give each a cache to allow reusing a given cache for multiple uses of the sample
    params = []
    # TODO: consider multiple covariances
    # TODO: need to find how to sample this in torch
    samples = torch.from_numpy(randState.multivariate_normal(lgc.parameters[0, :].numpy(), lgc.covariance[0], size=count))
    for i in range(count):
        print(f"Sample {i}", end='\r')
        params.append(samples[i, :].reshape(1, -1))
    logging.info(f"Finished monte carlo samples")
    return params


def _monteCarloFirstMoment(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams,
                           obsX: np.ndarray = None) -> torch.Tensor:
    """
    Computes a tensor of monte carlo samples for the first moment
    """
    values = torch.zeros(size=(len(MonteCarloParams),), dtype=torch.float)
    for i, param in enumerate(params):
        print(f"MC first moment {i}\r", end='\r')
        lgc.set_node_parameters(param)
        values[i] = Expectation(psdd, lgc, EVCache(), obsX).flatten()
    logging.info(f"Finished monte carlo first moment")
    return values


# TODO cleanup - this method is unused and should probably be removed
def monteCarloMeanAndParameterVariance(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams,
                                       obsX: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the variance over the parameters for the circuit using monte carlo
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param params:  List of parameters to use for the samples,
                    takes advantage of the fact we can reuse a cache for a given value
    @param obsX:    Observation vector
    @return  Tuple of mean and parameter variance
    """
    values = _monteCarloFirstMoment(psdd, lgc, params, obsX)
    return torch.mean(values), torch.var(values)


# TODO cleanup - this method is incorrect and should probably be removed
# is calculating monte carlo estimate of the second moment and monte carlo estimate of the first moment,
# instead of monte carlo estimate of the variance, no easy way to fix without removing mean parameter
def monteCarloInputVariance(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams, mean: torch.Tensor,
                            obsX: np.ndarray = None) -> torch.Tensor:
    """
    Computes the variance over the inputs for the circuit using monte carlo
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param params:  List of parameters to use for the samples,
                    takes advantage of the fact we can reuse a cache for a given value
    @param mean:  Previously computed mean, if excluded computes it
    @param obsX:    Observation vector
    @return  Input variance
    """
    # use passed mean if given, compute if not
    if mean is None:
        values = _monteCarloFirstMoment(psdd, lgc, params, obsX)
        mean = torch.mean(values)

    # monte carlo average of second moment
    values = torch.zeros(size=(len(params),), dtype=torch.float)
    for i, param in enumerate(params):
        print(f"MC input variance {i}", end='\r')
        lgc.set_node_parameters(param)
        # TODO: share cache with above method
        values[i] = moment(psdd, lgc, 2, EVCache(), obsX).flatten()
    logging.info(f"Finished monte carlo input variance")

    return torch.mean(values) - mean**2


def monteCarloPrediction(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams,
                         obsX: np.ndarray = None, prefix: str = '') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a prediction of the mean, parameter variance, and input variance of the circuit.
    Faster than calling the two other monte carlo methods if all three values are needed
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param params:  List of parameters to use for the samples,
                    takes advantage of the fact we can reuse a cache for a given value
    @param obsX:    Observation vector
    @param prefix:  Prefix for printing
    @return  Tuple of mean, parameter variance, and input variance
    """
    # collect samples for the first two moments, doing in a single loop saves effort setting the parameters
    inputs = obsX.shape[0]
    size = (inputs, len(params))
    firstMoments = torch.zeros(size=size, dtype=torch.float)
    secondMoments = torch.zeros(size=size, dtype=torch.float)
    for i, param in enumerate(params):
        print(f"Evaluating {prefix} MC sample {i}                       ", end='\r')
        lgc.set_node_parameters(param)
        cache = EVCache()
        firstMoments[:, i] = Expectation(psdd, lgc, cache, obsX)
        secondMoments[:, i] = moment(psdd, lgc, 2, cache, obsX)
    print(f"Finished {prefix} monte carlo predictions               ", end='\r')

    # E[M1(phi)]
    mean = torch.mean(firstMoments, dim=1)
    # mean, var[M1[(phi)], E[M2(phi) - M1(phi)^2]
    return mean,\
        torch.var(firstMoments, dim=1),\
        torch.mean(secondMoments - firstMoments ** 2, dim=1)


def _monteCarloIteration(psdd: PSddNode, lgc: BaseCircuit, param: torch.Tensor,
                         obsX: np.ndarray = None, prefix: str = '', i: int = -1
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single iteration for the parallel monte carlo prediction"""
    print(f"Evaluating {prefix} MC sample {i}                       ", end='\r')
    lgc = copy.deepcopy(lgc)
    lgc.set_node_parameters(param)
    cache = EVCache()
    firstMoment = Expectation(psdd, lgc, cache, obsX)
    secondMoment = moment(psdd, lgc, 2, cache, obsX)
    return firstMoment, secondMoment


def monteCarloPredictionParallel(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams,
                                 obsX: np.ndarray = None, prefix: str = '', jobs: int = -1
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a prediction of the mean, parameter variance, and input variance of the circuit.
    Faster than calling the two other monte carlo methods if all three values are needed
    @param psdd:    Probabilistic circuit root
    @param lgc:     Logistic or regression circuit
    @param params:  List of parameters to use for the samples,
                    takes advantage of the fact we can reuse a cache for a given value
    @param obsX:    Observation vector
    @param prefix:  Prefix for printing
    @param jobs:    Jobs to run in parallel
    @return  Tuple of mean, parameter variance, and input variance
    """
    # collect samples for the first two moments, doing in a single loop saves effort setting the parameters
    delayedFunc = delayed(_monteCarloIteration)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(psdd, lgc, param, obsX, prefix, i)
        for i, param in enumerate(params)
    )
    firstMoments, secondMoments = zip(*result)
    firstMoments = torch.concat(firstMoments, dim=1)
    secondMoments = torch.concat(secondMoments, dim=1)

    # E[M1(phi)], var[M1[(phi)], E[M2(phi) - M1(phi)^2]
    return torch.mean(firstMoments, dim=1),\
        torch.var(firstMoments, dim=1),\
        torch.mean(secondMoments - firstMoments ** 2, dim=1)


def deltaMeanAndParameterVariance(psdd: PSddNode, lgc: BaseCircuit, cache: EVCache,
                                  obsX: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the variance over the inputs for the circuit
    @param psdd:  Probabilistic circuit root
    @param lgc:   Logistic or regression circuit
    @param cache: Cache for the given observed value, need to recreate each time the mean is zeroed, which is currently
                  done on function call
    @param obsX:  Observed value to input
    @return  Tuple of mean, parameter variance
    """
    if len(lgc.covariance) == 0:
        raise ValueError("Circuit must have covariance to compute a mean and variance")

    lgc.zero_grad(True)
    # E[M1[phi]],
    mean = Expectation(psdd, lgc, cache, obsX)
    mean.backward(torch.ones((obsX.shape[0], lgc.num_classes)), inputs=lgc.parameters)
    grad = lgc.parameters.grad

    # var[M1[phi]], gradient vector is a row vector instead of a column vector
    variance = torch.mm(grad, torch.mm(torch.from_numpy(lgc.covariance[0]), grad.T))
    return mean.flatten(), variance


def deltaInputVariance(psdd: PSddNode, lgc: BaseCircuit, cache: EVCache, obsX: np.ndarray = None,
                       mean: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the variance over the inputs for the circuit using the delta method
    @param psdd:  Probabilistic circuit root
    @param lgc:   Logistic or regression circuit
    @param cache: Cache for the given observed value, need to recreate for a different input
    @param mean:  Previously computed mean, if excluded computes it
    @param obsX:  Observed value to input
    @return  Input variance
    """
    lgc.zero_grad(False)
    lgc.set_node_parameters(lgc.parameters)
    if mean is None:
        mean = Expectation(psdd, lgc, cache, obsX)
    secondMoment = moment(psdd, lgc, 2, cache, obsX)
    # E[M2(phi) - M1(phi)^2]
    return secondMoment - mean**2, mean.flatten()


def exactDeltaTotalVariance(psdd: PSddNode, lgc: BaseCircuit, cache: EVCache, obsX: np.ndarray = None,
                            mean: torch.Tensor = None, inputVar: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the total variance for the circuit using the exact delta method
    Note that after calling this function, the circuit parameters and the parameter mean may be out of sync
    @param psdd:  Probabilistic circuit root
    @param lgc:   Logistic or regression circuit
    @param cache: Cache for the given observed value, need to recreate for a different input
    @param mean:  Previously computed mean, if excluded computes it
    @param inputVar    Previously computed input variance from deltaInputVariance, if excluded computes it
    @param obsX:  Observed value to input
    @return  Input variance
    """
    # the difference between approx input variance and exact total variance is an extra hessian term
    # exact input variance also includes a negative parameter variance term, cancels out regular parameter variance
    if inputVar is None:
        inputVar, _ = deltaInputVariance(psdd, lgc, cache, obsX, mean)

    def hess_second_moment(params):
        lgc.set_node_parameters(params)
        return moment(psdd, lgc, 2, EVCache(), obsX)

    originalParams = lgc.parameters
    hess = torch.autograd.functional.hessian(hess_second_moment, originalParams)

    return inputVar + torch.trace(torch.mm(hess.squeeze(), torch.from_numpy(lgc.covariance[0])))


def _monteCarloGaussianInputIteration(lgc: BaseCircuit, param: Optional[torch.Tensor],
                                      inputMean: np.ndarray, inputCovariance: np.ndarray, inputSamples: int,
                                      inputReducer: callable, obsX: np.ndarray = None, prefix: str = '', i: int = -1,
                                      seed: int = 1337, randState: RandomState = None
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single iteration for the parallel monte carlo prediction"""
    if param is not None:
        print(f"Evaluating {prefix} MC sample {i}                       ", end='\r')
        lgc = copy.deepcopy(lgc)
        lgc.set_node_parameters(param)

    # the goal here is to process all test samples and all input samples in one large batch of size test * input
    # start by constructing a 3D matrix of test sample * input sample * feature
    testSamples = obsX.shape[0]
    features = obsX.shape[1]
    obsXAugmented = np.zeros((testSamples, inputSamples, features))
    if randState is None:
        randState = RandomState(seed)
    for sample in range(testSamples):
        image = obsX[sample, :]
        missingIndexes = image == -1

        obsXAugmented[sample, :, :] = np.repeat(image.reshape(1, 1, -1), repeats=inputSamples, axis=1)
        # If no missing indexes, no need to handle samples
        if missingIndexes.sum() != 0:
            # Need to sample the the distribution the requested number of times, then replace all -1 values
            missingMean, missingCov = inputReducer(image, inputMean, inputCovariance)
            missingSamples = randState.multivariate_normal(missingMean, missingCov, size=inputSamples)
            # TODO: the following clamp is wrong and I should feel ashamed for writing it, but right now I need to test other stuff
            obsXAugmented[sample, :, missingIndexes] = np.clip(missingSamples.T, 0, 1)

        # we should have filled in all -1 values in the final array
        assert (obsXAugmented[sample, :, :] == -1).sum() == 0

    # now we have a feature * test sample * input sample array, but to evaluate the circuit we need feature * sample
    obsXInput = obsXAugmented.reshape(testSamples * inputSamples, features)
    # for i in range(inputSamples):
    #     if (obsXAugmented[0, i, :] != obsXInput[i]).sum() != 0:
    #         print("Vector reshape failed")
    #     if (obsXAugmented[1, i, :] != obsXInput[inputSamples + i]).sum() != 0:
    #         print("Vector reshape failed")
    features = lgc.calculate_features(obsXInput)

    # Process the regression circuit values to get means
    # reshape so we have means per test sample
    predictions = torch.mm(torch.from_numpy(features), lgc.parameters.T).reshape(testSamples, inputSamples)

    # average over dimension 1 to get mean and covariances to return
    return torch.mean(predictions, dim=1).reshape(-1, 1),\
        torch.var(predictions, dim=1).reshape(-1, 1)


def monteCarloGaussianInputOnly(lgc: BaseCircuit, inputMean: np.ndarray, inputCovariance: np.ndarray, inputSamples: int,
                                inputReducer: callable = conditionalGaussian, randState: RandomState = None,
                                obsX: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a prediction of the mean, parameter variance, and input variance of the circuit using the gaussian method
    for handling input variances.
    @param lgc:              Logistic or regression circuit
    @param inputMean:        Mean vector of the input distribution
    @param inputCovariance:  Covariance matrix of the input distribution
    @param inputSamples:     Number of samples to take from the input distribution
    @param inputReducer:     Function to reduce the random variables, typically marginal or conditional
    @param randState:        Random state for sampling inputs
    @param obsX:             Observation vector
    @return  Tuple of mean, and input variance
    """
    # to ensure consistency despite the fact we are generating random numbers on threads
    # we generate a random seed for each thread to use for its random number generation
    if randState is None:
        randState = RandomState(1337)

    means, covariances = _monteCarloGaussianInputIteration(
        lgc, None, inputMean, inputCovariance, inputSamples, inputReducer, obsX, randState=randState
    )

    # mean, parameter variance (averaged across input), input variance (averaged across parameters)
    return means.squeeze(), covariances.squeeze()


def monteCarloGaussianParamAndInput(lgc: BaseCircuit, params: MonteCarloParams,
                                    inputMean: np.ndarray, inputCovariance: np.ndarray, inputSamples: int,
                                    inputReducer: callable = conditionalGaussian, randState: RandomState = None,
                                    obsX: np.ndarray = None,
                                    prefix: str = '', jobs: int = -1
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a prediction of the mean, parameter variance, and input variance of the circuit using the gaussian method
    for handling input variances.
    @param lgc:              Logistic or regression circuit
    @param params:           List of parameters to use for the samples
    @param inputMean:        Mean vector of the input distribution
    @param inputCovariance:  Covariance matrix of the input distribution
    @param inputSamples:     Number of samples to take from the input distribution
    @param inputReducer:     Function to reduce the random variables, typically marginal or conditional
    @param randState:        Random state for sampling inputs
    @param obsX:             Observation vector
    @param prefix:           Prefix for printing
    @param jobs:             Jobs to run in parallel
    @return  Tuple of mean, parameter variance, and input variance
    """
    # to ensure consistency despite the fact we are generating random numbers on threads
    # we generate a random seed for each thread to use for its random number generation
    if randState is None:
        randState = RandomState(1337)
    # collect samples for the first two moments, doing in a single loop saves effort setting the parameters
    delayedFunc = delayed(_monteCarloGaussianInputIteration)
    result = Parallel(n_jobs=jobs,
                      **_joblib_parallel_args(prefer='processes'))(
        delayedFunc(lgc, param, inputMean, inputCovariance, inputSamples, inputReducer, obsX, prefix, i,
                    randState.randint(0, 2**32 - 1))
        for i, param in enumerate(params)
    )
    means, covariances = zip(*result)
    means = torch.concat(means, dim=1)
    covariances = torch.concat(covariances, dim=1)

    # mean, parameter variance (averaged across input), input variance (averaged across parameters)
    return torch.mean(means, dim=1),\
        torch.var(means, dim=1),\
        torch.mean(covariances, dim=1)
