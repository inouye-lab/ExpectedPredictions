import numpy as np
import torch
from typing import Tuple, List

from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from circuit_expect import Expectation, moment
from numpy.random import RandomState
from pypsdd import PSddNode


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
    print(f"Finished monte carlo samples")
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
    print(f"Finished monte carlo first moment")
    return values


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
    return torch.mean(values), torch.std(values)


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
    print(f"Finished monte carlo input variance")

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
    firstMoments = torch.zeros(size=(len(params),), dtype=torch.float)
    secondMoments = torch.zeros(size=(len(params),), dtype=torch.float)
    for i, param in enumerate(params):
        print(f"Evaluating {prefix} MC sample {i}", end='\r')
        lgc.set_node_parameters(param)
        cache = EVCache()
        firstMoments[i] = Expectation(psdd, lgc, cache, obsX).flatten()
        secondMoments[i] = moment(psdd, lgc, 2, cache, obsX).flatten()
    print(f"Finished {prefix} monte carlo predictions", end='\r')

    # E[M1(phi)]
    mean = torch.mean(firstMoments)
    # mean, var[M1[(phi)], E[M2(phi) - M1(phi)^2]
    return mean, torch.std(firstMoments), torch.mean(secondMoments) - mean ** 2


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
                       mean: torch.Tensor = None) -> torch.Tensor:
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
    if mean is None:
        mean = Expectation(psdd, lgc, cache, obsX)
    secondMoment = moment(psdd, lgc, 2, cache, obsX)
    # E[M2(phi) - M1(phi)^2]
    return secondMoment - mean**2


def exactInputVariance(psdd: PSddNode, lgc: BaseCircuit, cache: EVCache, obsX: np.ndarray = None,
                       mean: torch.Tensor = None, var: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the variance over the inputs for the circuit using the exact method
    @param psdd:  Probabilistic circuit root
    @param lgc:   Logistic or regression circuit
    @param cache: Cache for the given observed value, need to recreate for a different input
    @param mean:  Previously computed mean, if excluded computes it
    @param var    Previously computed variance, if excluded computes it
    @param obsX:  Observed value to input
    @return  Input variance
    """
    raise Exception("Not yet implemented")
