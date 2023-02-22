import copy
import numpy as np
import torch
from typing import Tuple, List, Optional, Union, Any

from torch import Tensor

import uncertainty_baseline
from EVCache import EVCache
from LogisticCircuit.algo.BaseCircuit import BaseCircuit
from LogisticCircuit.util.DataSet import DataSet
from pypsdd import PSddNode
from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, \
    monteCarloPrediction, MonteCarloParams, monteCarloPredictionParallel, exactDeltaTotalVariance
from torch.distributions.normal import Normal

import logging
import gc
from sklearn.linear_model._logistic import (_joblib_parallel_args)
from scipy.stats import norm
from joblib import Parallel, delayed

SummaryFunction = callable
"""
  Function to use to generate summary

  @param dataset             The dataset used for the experiment
  @param mean                Tensor of averages with a value per sample in the dataset
  @param inputVariances      Tensor of input variances with a value per sample in the dataset
  @param parameterVariances  Tensor of parameter variances with a value per sample in the dataset
  @param totalVariances      Tensor of total variances with a value per sample in the dataset
  @return  Any desired summary statistic
"""

SummaryType = Tuple[
    float,  # MSE
    float, float, float, float,  # LL
    float, float, float,  # Var
    float, float,         # CE
    Tensor, Tensor, Tensor, Tensor, Tensor
]
"""
Result of an experiment, contains:
  Total error,
  avg input LL, avg param LL, avg total LL,
  avg input var, avg param var, avg total var,
  full mean vector, full input var vector, full param var vector,
  full p-value vector ignoring residual, full p-value vector
"""


def orElse(value: Optional[Any], fallback):
    if value is None:
        return fallback
    return value


def _gaussianLogLikelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood for a gaussian distribution"""
    clampVar = var.clamp(min=1e-10)
    return -0.5 * torch.log(torch.mul(2 * torch.pi, clampVar))\
        - 0.5 / clampVar * ((x - mean) ** 2)


def _gaussianPValue(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Computes the p-values of the true value for all samples in the mean and variance vectors"""
    normal = Normal(mean, torch.sqrt(var.clamp(min=1e-10)))
    return 2 * normal.cdf(mean - torch.abs(mean - x))


def _confidenceError(pValues: torch.Tensor) -> torch.Tensor:
    """Computes the error between the ideal confidence values and the true one"""
    sortedValues, _ = torch.sort(1 - pValues)
    # the linspace represents the ideal value at each index, while sorted values are the actual value we got
    return torch.mean(torch.abs(sortedValues - torch.linspace(0, 1, pValues.shape[0])))


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
    error = torch.pow(mean - dataset.labels, 2)
    inputLikelihood = _gaussianLogLikelihood(dataset.labels, mean, inputVariances)
    parameterLikelihood = _gaussianLogLikelihood(dataset.labels, mean, parameterVariances)
    noResidualLikelihood = _gaussianLogLikelihood(dataset.labels, mean, inputVariances + parameterVariances)
    totalLikelihood = _gaussianLogLikelihood(dataset.labels, mean, totalVariances)
    noResidualPValues = _gaussianPValue(dataset.labels, mean, inputVariances + parameterVariances)
    totalPValues = _gaussianPValue(dataset.labels, mean, totalVariances)

    return torch.mean(error).item(), torch.mean(inputLikelihood).item(), torch.mean(parameterLikelihood).item(),\
        torch.mean(noResidualLikelihood).item(), torch.mean(totalLikelihood).item(), \
        torch.mean(inputVariances).item(), torch.mean(parameterVariances).item(), torch.mean(totalVariances).item(),\
           _confidenceError(noResidualPValues).item(), _confidenceError(totalPValues).item(), \
        mean, inputVariances, parameterVariances, noResidualPValues, totalPValues


def computeConfidenceResidualUncertainty(confidence: float) -> SummaryFunction:
    """
    Creates a summary function that computes the residual uncertainty for the given percentage
    @param confidence  Confidence percent, used for selecting the uncertainty. Higher percents means larger intervals, meaning more values are accepted
    """
    zValue = norm.ppf((confidence + 1) / 2)
    logging.info("Computed zValue for {}% confidence is {}".format(confidence * 100, zValue))

    # noinspection PyUnusedLocal
    # Required to meet the signature for this function
    def summaryFunction(dataset: DataSet, mean: torch.Tensor, inputVariances: torch.Tensor,
                        parameterVariances: torch.Tensor, totalVariances: torch.Tensor) -> float:
        residual, _ = torch.sort(torch.abs(mean - dataset.labels) / zValue - torch.sqrt(totalVariances))
        # if the value is negative, it is inside the confidence interval, while positive is outside
        size = residual.shape[0]

        # travel confidence% of the way through the array, since these are sorted smallest to largest that will bw a
        # number such that confidence% of the array is smaller
        selectedIndex = max(0, min(size - 1, int(size * confidence)))
        selectedResidual = residual[selectedIndex].item()
        # noinspection PyTypeChecker
        # ^ it returns a tensor of booleans
        logging.info("Confidence changed from {}% to {}% using the residual {}".format(
            torch.count_nonzero(residual < 0) * 100 / size,
            torch.count_nonzero((residual - selectedResidual) < 0) * 100 / size,
            selectedResidual
        ))

        # most of our math works in terms of variance instead of STD, so we need to square this
        # however, we also need to ensure we keep the sign, did we overestimate or underestimate?
        # TODO: should we instead square after summing the two? so (sqrt(totalVar) + residual)^2 ?
        return selectedResidual * abs(selectedResidual)
    return summaryFunction


# noinspection PyUnusedLocal
# Required to meet the signature for this function
def computeMSEResidualUncertainty(dataset: DataSet, mean: torch.Tensor, inputVariances: torch.Tensor,
                                  parameterVariances: torch.Tensor, totalVariances: torch.Tensor) -> float:
    """
    Computes residual uncertainty using the MSE method, which is an approximation of the MLE estimator of residual
    """
    return torch.mean(torch.pow(mean - dataset.labels, 2) - totalVariances).item()


# noinspection PyUnusedLocal
# Required to meet the signature for this function
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


def monteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams, dataset: DataSet,
                                    jobs: int = -1, summaryFunction: SummaryFunction = None,
                                    residualUncertainty: float = 0) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset, used for time benchmark
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param params:              Parameters to use for estimates
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, inputVariances, parameterVariances = _baseParallelOverSamples(
        psdd, lgc, dataset, jobs, _monteCarloIteration, params
    )
    totalVariances = parameterVariances + inputVariances + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(dataset, mean, inputVariances, parameterVariances, totalVariances)


def fastMonteCarloGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, params: MonteCarloParams, dataset: DataSet,
                                        jobs: int = -1, summaryFunction: SummaryFunction = None,
                                        residualUncertainty: float = 0) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset,
    performing parallel over parameters and using torch to do parallel over inputs
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param params:              Parameters to use for estimates
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # prepare parallel
    mean, parameterVariances, inputVariances = monteCarloPredictionParallel(
        psdd, lgc, params, dataset.images, jobs=jobs
    )
    totalVariances = parameterVariances + inputVariances + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(dataset, mean, inputVariances, parameterVariances, totalVariances)


# noinspection PyUnusedLocal
# Required to meet the signature for this function
def _deltaIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int,
                    computeInput: bool = False) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Evaluates a single iteration of the delta method, used for parallel"""
    print(f"Evaluating delta method for sample {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    # clone the circuit to prevent conflicting with the other threads
    lgc = copy.deepcopy(lgc)
    lgc.set_node_parameters(lgc.parameters.detach(), set_circuit=True, set_require_grad=True)

    mean, sampleParamVar = deltaMeanAndParameterVariance(psdd, lgc, cache, feature)
    if computeInput:
        inputVar, _ = deltaInputVariance(psdd, lgc, cache, feature, mean=mean)
        return mean, sampleParamVar, inputVar
    return mean, sampleParamVar


def deltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1,
                               summaryFunction: SummaryFunction = None, residualUncertainty: float = 0) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _deltaIteration)

    # can easily batch input variance as we are not taking gradients there
    inputVariance, _ = deltaInputVariance(psdd, lgc, EVCache(), dataset.images)
    # add variances directly
    inputVariance = torch.clamp(inputVariance, min=0).squeeze()
    totalVariance = inputVariance + paramVariance + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(dataset, mean, inputVariance, paramVariance, totalVariance)


# noinspection PyUnusedLocal
# Required to meet the signature for this function
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


def exactDeltaGaussianLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1,
                                    summaryFunction: SummaryFunction = None, residualUncertainty: float = 0
                                    ) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # residual uncertainty is added in later as we don't want that included as part of input or parameter
    # thus, this variable has a bit of an odd name as we cannot canonically return total uncertainty
    mean, paramVariance, inputParameterVariance = _baseParallelOverSamples(
        psdd, lgc, dataset, jobs, _exactDeltaIteration
    )
    inputVariance = inputParameterVariance - paramVariance

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputVariance, paramVariance, inputParameterVariance + residualUncertainty
    )


def monteCarloParamLogLikelihood(trainingSampleMean: np.ndarray, lgc: BaseCircuit, params: MonteCarloParams,
                                 dataset: DataSet, jobs = -1, summaryFunction: SummaryFunction = None,
                                 residualUncertainty: float = 0) -> SummaryType:
    """
    Baseline function to test the parameter variance using the monte carlo method without considering input variance.
    Missing values are handled through a training sample mean.

    @param trainingSampleMean:  Average value of each feature in the training data
    @param lgc:                 Circuit instance
    @param dataset:             Dataset for computing the full value
    @param params:              Parameters to use for estimates
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # prepare parallel
    mean, parameterVariances = uncertainty_baseline.monteCarloPredictionParallel(
        trainingSampleMean, lgc, params, dataset.images, prefix='Baseline Param', jobs=jobs
    )

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, torch.zeros(size=mean.shape, dtype=torch.float), parameterVariances,
        parameterVariances + residualUncertainty
    )


# noinspection PyUnusedLocal
# Required to meet the signature for this function
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


def deltaParamLogLikelihood(trainingSampleMean: np.ndarray, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1,
                            summaryFunction: SummaryFunction = None, residualUncertainty: float = 0) -> SummaryType:
    """
    Baseline function to test the parameter variance using the delta method without considering input variance.
    Missing values are handled through a training sample mean.

    @param trainingSampleMean:  Average value of each feature in the training data
    @param lgc:                 Circuit instance
    @param dataset:             Dataset for computing the full value
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance = _baseParallelOverSamples(None, lgc, dataset, jobs, _deltaParamIteration, trainingSampleMean)

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, torch.zeros(size=mean.shape, dtype=torch.float), paramVariance,
        paramVariance + residualUncertainty
    )


def inputLogLikelihood(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, summaryFunction: SummaryFunction = None,
                       residualUncertainty: float = 0) -> SummaryType:
    """
    Computes likelihood and input variances over the entire dataset
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    # delta method for input variance is already essentially the same as only considering input variance
    # difference is just the training method (and the fact delta normally gets param variance)
    inputVariance, mean = deltaInputVariance(psdd, lgc, EVCache(), dataset.images)
    inputVariance = torch.clamp(inputVariance, min=0).squeeze()

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputVariance, torch.zeros(size=mean.shape, dtype=torch.float),
        inputVariance + residualUncertainty
    )


def deltaGaussianLogLikelihoodBenchmarkTime(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1,
                                            summaryFunction: SummaryFunction = None, residualUncertainty: float = 0
                                            ) -> SummaryType:
    """
    Computes likelihood and variances over the entire dataset
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, paramVariance, inputVariance = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _deltaIteration, True)
    inputVariance = torch.clamp(inputVariance, min=0)
    totalVariance = inputVariance + paramVariance + residualUncertainty

    gc.collect()
    return orElse(summaryFunction, _summarize)(dataset, mean, inputVariance, paramVariance, totalVariance)


def _baselineInputIteration(psdd: PSddNode, lgc: BaseCircuit, feature: np.ndarray, y: torch.Tensor, i: int
                            ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Evaluates a single iteration of the baseline input variance method, used for parallel"""
    print(f"Evaluating input baseline method for sample {i}", end='\r')
    cache = EVCache()
    feature = feature.reshape(1, -1)
    inputVar, mean = deltaInputVariance(psdd, lgc, cache, feature)
    return mean, inputVar


def inputLogLikelihoodBenchmarkTime(psdd: PSddNode, lgc: BaseCircuit, dataset: DataSet, jobs: int = -1,
                                    summaryFunction: SummaryFunction = None, residualUncertainty: float = 0
                                    ) -> SummaryType:
    """
    Computes likelihood and input variances over the entire dataset.
    Unlike inputLogLikelihood, foregoes batching to make it more comparable to other methods that don't support batching
    @param psdd:                Probabilistic circuit root
    @param lgc:                 Logistic or regression circuit
    @param dataset:             Dataset for computing the full value
    @param jobs:                Max number of parallel jobs to run, use -1 to use the max possible
    @param summaryFunction:     Function to use to generate the summary
    @param residualUncertainty: Uncertainty from sources other than input and parameters, summed into final total
    @return  Tuple of total error, average input LL, average param LL, average total LL,
             average input variance, average param variance, average total variance
    """
    mean, inputVariance = _baseParallelOverSamples(psdd, lgc, dataset, jobs, _baselineInputIteration)

    gc.collect()
    return orElse(summaryFunction, _summarize)(
        dataset, mean, inputVariance, torch.zeros(size=mean.shape, dtype=torch.float),
        inputVariance + residualUncertainty
    )
