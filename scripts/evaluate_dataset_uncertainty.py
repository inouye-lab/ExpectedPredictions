import json
import math
import sys

import csv
import gc

import os

import logging
from datetime import datetime

from torch import Tensor

sys.path.append('.')

import argparse
from typing import Tuple, List, Optional

import gzip
import pickle

import numpy as np
from numpy.random import RandomState
import pypsdd.psdd_io

from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree
from LogisticCircuit.util.DataSet import DataSet
from LogisticCircuit.algo.NeuralNetworkBaseline import NeuralNetworkRegressor

from pypsdd.vtree import Vtree as PSDD_Vtree
from pypsdd.manager import PSddManager

from uncertainty_utils import meanImputation, conditionalMeanImputation, conditionalGaussian, marginalizeGaussian, \
    psddMpeImputation
from uncertainty_calculations import sampleMonteCarloParameters
from uncertainty_validation import deltaGaussianLogLikelihood, monteCarloGaussianLogLikelihood, \
    fastMonteCarloGaussianLogLikelihood, exactDeltaGaussianLogLikelihood, monteCarloParamLogLikelihood, \
    deltaParamLogLikelihood, inputLogLikelihood, SummaryType, deltaGaussianLogLikelihoodBenchmarkTime, \
    inputLogLikelihoodBenchmarkTime, computeConfidenceResidualUncertainty, basicExpectation, basicImputation, \
    computeMSEResidualUncertainty, deltaNoInputLogLikelihood, residualPerSampleInput, \
    monteCarloGaussianInputOnlyLogLikelihood, monteCarloGaussianParamInputLogLikelihood, \
    monteCarloPSDDInputOnlyLogLikelihood, monteCarloPSDDParamInputLogLikelihood
from uncertainty_neural_network import basicNNImputation, monteCarloGaussianNNInputOnlyLogLikelihood, \
    monteCarloPSDDNNInputOnlyLogLikelihood

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


class Result:
    """Represents a single result from either method"""
    method: str
    trainPercent: float
    missingPercent: float
    runtime: Optional[float]
    residualRuntime: Optional[float]
    totalError: float

    inputLL: float
    paramLL: float
    noResidualLL: float
    totalLL: float

    inputVar: float
    paramVar: float
    totalVar: float
    residualUncertainty: float

    ceNoResidual: float
    ceResidual: float

    def __init__(self, method: str, trainPercent: float, missingPercent: float, residualUncertainty: float,
                 totalError: float,
                 inputLL: float, paramLL: float, noResidualLL: float, totalLL: float,
                 inputVar: float, paramVar: float, totalVar: float,
                 ceNoResidual: float, ceResidual: float):
        self.method = method
        self.trainPercent = trainPercent
        self.missingPercent = missingPercent
        self.totalError = totalError

        self.inputLL = inputLL
        self.paramLL = paramLL
        self.noResidualLL = noResidualLL
        self.totalLL = totalLL

        self.inputVar = inputVar
        self.paramVar = paramVar
        self.totalVar = totalVar
        self.residualUncertainty = residualUncertainty

        self.ceNoResidual = ceNoResidual
        self.ceResidual = ceResidual

        self.runtime = None

    def print(self):
        logging.info(f"{self.method} @ train {self.trainPercent}, missing {self.missingPercent}")
        logging.info(f"    error: {self.totalError}")
        logging.info(f"    confidence error: w/o residual {self.ceNoResidual}, w/ residual {self.ceResidual}")
        logging.info(f"    ll: input {self.inputLL}, param {self.paramLL}, input+param {self.noResidualLL}, total {self.totalLL}")
        logging.info(f"    var: input {self.inputVar}, param {self.paramVar}, total {self.totalVar}")

    def getResultRow(self):
        return [
            self.method, self.trainPercent, self.missingPercent,
            self.runtime, self.totalError,
            self.inputLL, self.paramLL, self.noResidualLL, self.totalLL,
            self.inputVar, self.paramVar, self.residualUncertainty, self.totalVar,
            self.ceNoResidual, self.ceResidual
        ]


if __name__ == '__main__':
    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('model', type=str, help='Model to use for expectations')
    parser.add_argument('--prefix', type=str, default='', help='Folder prefix for both the model and the output')
    parser.add_argument('--nn_folder', type=str, default='', help='Location of the neural network baseline. If unset, skips')
    parser.add_argument("--classes", type=int, required=True, help="Number of classes in the dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--fmap", type=str, default='', help="Path to the dataset fmap")
    parser.add_argument("--retrain_dir", type=str, required=True, help="Location of folders for retrained models")

    # Output configuration
    parser.add_argument('--output', type=str, help='Location for result csv')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Verbosity level')
    parser.add_argument("--log_results",  action='store_true',
                        help="If set, results of the script are logged in tabular form after the script finishes "
                             "executing. "
                             "Redundant to the CSV file and requires more memory, but useful when running in an IDE.")

    # Methods to include
    parser.add_argument("--skip_delta",  action='store_true',
                        help="If set, the delta method is skipped, running just MC")
    parser.add_argument("--exact_delta",  action='store_true',
                        help="If set, runs the exact delta method")
    parser.add_argument("--parameter_baseline",  action='store_true',
                        help="If set, runs the baseline parameter uncertainty using the dataset mean")
    parser.add_argument("--input_baseline",  action='store_true',
                        help="If set, runs the baseline input uncertainty using the parameter mean")
    parser.add_argument("--samples", type=int, default=0,
                        help="Number of monte carlo samples for parameters")
    parser.add_argument("--include_trivial",  action='store_true',
                        help="If set, runs the trivial methods that do not compute uncertainty, only residual")
    parser.add_argument("--include_residual_input",  action='store_true',
                        help="If set, runs the residual input method for trivial methods")
    parser.add_argument("--input_samples", type=int, default=0,
                        help="Number of monte carlo samples on the input distribution for missing values")
    parser.add_argument("--psdd_samples", type=int, default=0,
                        help="Number of monte carlo samples on the psdd distribution for missing values")
    parser.add_argument("--skip_mc",  action='store_true',
                        help="If set, skips the main monte carlo method even when parameter samples is set")

    parser.add_argument("--benchmark_time",  action='store_true',
                        help="If set, disables batching on several methods to make the times more comparable")

    # Experiment configuration
    parser.add_argument("--missing", type=float, nargs='*', help="Percent of data to treat as missing")
    parser.add_argument("--data_percents", type=float, nargs='*', help="Percentages of the dataset to use in training")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for dataset selection")
    parser.add_argument("--global_missing_features",  action='store_true',
                        help="If set, the same feature will be missing in all samples. "
                             "If unset, each sample will have missing features selected separately")
    parser.add_argument("--evaluate_validation",  action='store_true',
                        help="If set, evaluates the validation dataset instead of the testing dataset. "
                             "Used to validate against known data.")
    parser.add_argument("--skip_enforce_boolean",  action='store_true',
                        help="If set, runs the likely wrong behavior of not enforcing booleans.")
    parser.add_argument("--full_training_gaussian",  action='store_true',
                        help="If set, uses the full training set for gaussian instead of the set used in RC training. "
                             "Provides more parity with PSDD training.")

    # Residual configuration
    parser.add_argument("--mse_residual",  action='store_true',
                        help="If set, uses the MSE method to compute residual uncertainty")
    parser.add_argument("--conformal_confidence", type=float, default=-1,
                        help="Confidence level to use for conformal prediction")
    parser.add_argument("--residual_missingness",  action='store_true',
                        help="If set, uses missing values for residual. If unset, residual is calculated without missing values")

    #
    # parsing the args
    args = parser.parse_args()

    # setup logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()

    # to file
    log_dir = os.path.join(args.output, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    date_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler("{0}/{1}-{2}.log".format(log_dir, args.model, date_string))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # and to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    #
    # setting verbosity level
    if args.verbose == 1:
        root_logger.setLevel(logging.INFO)
    elif args.verbose == 2:
        root_logger.setLevel(logging.DEBUG)

    # Print welcome message
    outputName = args.model + "-" + date_string
    args_out_path = os.path.join(log_dir, outputName + '.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    # Now, to the script
    FOLDER = args.prefix + args.model
    VTREE_FILE = FOLDER + ".vtree"
    GLC_FILE = FOLDER + ".glc"
    PSDD_FILE = FOLDER + ".psdd"

    lc_vtree = LC_Vtree.read(VTREE_FILE)

    logging.info("Loading samples...")
    with gzip.open(args.data, 'rb') as f:
        rawData = pickle.load(f)
    (trainingImages, trainingLabels), (validImages, validLabels), (testImages, testLabels) = rawData
    # TODO: probably worth deleting these, was just for backwards compat with old experiments
    if args.evaluate_validation:
        testImages = validImages
        testLabels = validLabels

    logging.info("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    # Stat datasets
    validSamples = validImages.shape[0]
    testSamples = testImages.shape[0]
    variables = testImages.shape[1]
    logging.info("Running {} dataset with {} validation and {} testing samples and {} variables".format(
        args.model, validSamples, testSamples, variables
    ))

    # populate missing datasets
    logging.info("Preparing missing datasets")
    randState = RandomState(args.seed)

    # if given a fmap, handle missing percentages as features in the map
    if args.fmap != '':
        with open(args.fmap, "rb") as f:
            fmap = np.array(pickle.load(f))
        uniqueIndexes = np.unique(fmap)
        uniqueVariables = len(uniqueIndexes)

        def sampleMissing(percent: float) -> np.ndarray:
            return np.isin(
                fmap, randState.choice(uniqueIndexes, size=math.floor(uniqueVariables * percent), replace=False)
            )
    else:
        # without the fmap, treat each variable as unique values, could lead to a higher covariance than expected
        def sampleMissing(percent: float) -> np.ndarray:
            return randState.choice(variables, size=math.floor(variables * percent), replace=False)

    testSets: List[Tuple[float, DataSet, Optional[DataSet]]] = []
    for missing in args.missing:
        missingTestImages = np.copy(testImages)
        # -1 is an internal value representing missing
        if args.global_missing_features:
            missingTestImages[:, sampleMissing(missing)] = -1
        else:
            for i in range(testSamples):
                missingTestImages[i, sampleMissing(missing)] = -1

        # only generate valid datasets if needed
        missingValidImages = None
        if args.residual_missingness:
            missingValidImages = np.copy(validImages)
            if args.global_missing_features:
                missingValidImages[:, sampleMissing(missing)] = -1
            else:
                for i in range(validSamples):
                    missingValidImages[i, sampleMissing(missing)] = -1
            missingValidImages = DataSet(missingValidImages, validLabels, one_hot = False)

        testSets.append((missing, DataSet(missingTestImages, testLabels, one_hot = False), missingValidImages))
    pureValidSet = DataSet(validImages, validLabels, one_hot=False)

    # first loop is over percents
    results: List[Result] = []

    # start CSV files for output
    resultsSummaryFile = open(os.path.join(args.output, outputName + "-summary.csv"), 'w')
    allResultsFile = open(os.path.join(args.output, outputName + "-all.csv"), 'w')
    resultsSummary = csv.writer(resultsSummaryFile)
    allResults = csv.writer(allResultsFile)
    csvHeaders = [
        "Name", "Train Percent", "Missing Percent",
        "Runtime", "MSE",
        "Input LL", "Param LL", "Input+Param LL", "Total LL",
        "Input Var", "Param Var", "Residual", "Total Var",
        "CE w/o residual", "CE w/ residual"
    ]
    resultsSummary.writerow(csvHeaders)
    allResults.writerow([
        "Name", "Train Percent", "Missing Percent", "Sample Index",
        "Expected", "Mean", "Input Variance", "Parameter Variance",
        "p-Value w/o Residual", "p-Value w/ Residual"
    ])
    resultsSummaryFile.flush()
    allResultsFile.flush()

    # build conformal summary function if requested
    # we may want n% of the dataset to be at best the computed number
    residualUncertaintyFunction: Optional[callable] = None
    if args.mse_residual:
        residualUncertaintyFunction = computeMSEResidualUncertainty
    elif args.conformal_confidence >= 0:
        residualUncertaintyFunction = computeConfidenceResidualUncertainty(args.conformal_confidence)

    # all experiments follow a general form, so abstract that out a bit
    # Argument signature: experiment_function(*experiment_arguments, testSet)
    def run_experiment(name: str, trainPercent: float, experiment_function, *experiment_arguments,
                       zero_grad: bool = False):
        for (missing, testSet, validSet) in testSets:
            testSet: DataSet
            validSet: Optional[DataSet]
            if zero_grad:
                lgc.zero_grad(True)

            # print experiment header
            logging.info("{}: Running {} at {}% training, {}% missing".format(
                args.model, name, trainPercent * 100, missing * 100
            ))

            # find residual uncertainty
            residualUncertainty: float = 0
            residualRuntime = 0
            if residualUncertaintyFunction is not None:
                dataset = validSet if validSet is not None else pureValidSet
                start_t = perf_counter()
                residualUncertainty = experiment_function(
                    *experiment_arguments, dataset=dataset, summaryFunction=residualUncertaintyFunction
                )
                end_t = perf_counter()
                residualRuntime = end_t - start_t
                logging.info("{}: {} residual uncertainty at {}% training and {}% missing: {}. Took {}".format(
                    args.model, name, percent * 100, missing * 100, residualUncertainty, residualRuntime
                ))

            # run experiment
            start_t = perf_counter()
            experiment_result: SummaryType = experiment_function(
                *experiment_arguments, dataset=testSet, residualUncertainty=residualUncertainty
            )
            result = Result(name, trainPercent, missing, residualUncertainty, *experiment_result[0:10])
            end_t = perf_counter()
            result.runtime = end_t - start_t
            result.residualRuntime = residualRuntime
            if args.log_results:
                results.append(result)

            # save to CSV file
            resultsSummary.writerow(result.getResultRow())
            resultsSummaryFile.flush()
            # save full results
            mean, inputVariance, parameterVariance, pValuesNoResidual, pValues = experiment_result[10:15]
            length = len(testSet.labels)
            assert len(mean) == length
            assert len(inputVariance) == length
            assert len(parameterVariance) == length
            assert len(pValuesNoResidual) == length
            assert len(pValues) == length
            for sample in range(length):
                allResults.writerow([
                    name, trainPercent, missing, sample,
                    testSet.labels[sample].item(),
                    mean[sample].item(),
                    inputVariance[sample].item(),
                    parameterVariance[sample].item(),
                    pValuesNoResidual[sample].item(),
                    pValues[sample].item()
                ])
            allResultsFile.flush()

            # print progress
            result.print()
            logging.info("{}: Uncertainty calculations {} at {}% training and {}% missing took {}"
                         .format(args.model, name, percent * 100, missing * 100, result.runtime))
            logging.info("----------------------------------------------------------------------------------------")

    # main experiment loop
    retrainFolder = args.prefix + args.retrain_dir
    for percent in args.data_percents:
        logging.info("Running {} percent".format(percent*100))
        logging.info("========================================================================================")
        percentFolder = retrainFolder + str(percent*100)

        # Always load the regression circuit
        with open(percentFolder + "percent.glc", 'r') as circuit_file:
            requireGrad = not args.skip_delta or args.exact_delta
            lgc = LogisticCircuit(lc_vtree, args.classes, circuit_file=circuit_file, requires_grad=requireGrad)

        # Load the neural network if asked
        nn: Optional[NeuralNetworkRegressor] = None
        if args.nn_folder != '':
            with open(args.nn_folder + str(percent*100) + "percent.pickle", 'rb') as network_file:
                nn = pickle.load(network_file)

        # training samples are used to construct training sample matrixes
        # for simplicity and consistency, we use the same samples in both, this is a parameter during training
        with open(percentFolder + "percentSamples.txt", 'r') as sample_file:
            indices = [int(v.strip()) for v in sample_file.readlines()]

            trainingData = DataSet(
                np.concatenate((trainingImages, validImages), axis=0)[indices, :],
                np.concatenate((trainingLabels, validLabels), axis=0)[indices],
                one_hot=False)
            trainingData.features = lgc.calculate_features(trainingData.images)
            mse = lgc.calculate_error(trainingData)
            logging.info("MSE for {} percent: {}".format(percent*100, mse))
            logging.info("----------------------------------------------------------------------------------------")

        # sample the training sample mean from the same set of images we use for evaluation
        # will be a bit more accurate to what we can actually produce as far as missing value imputation
        if args.parameter_baseline or args.include_trivial or args.include_residual_input or args.input_samples > 0:
            if args.full_training_gaussian:
                trainingSampleMean = np.mean(trainingImages, axis=0)
            else:
                trainingSampleMean = np.mean(trainingData.images, axis=0)

        # For the guassian methods, we also need a covariance matrix, will use the training sample mean with that
        if args.input_samples > 0 or args.include_trivial or args.include_residual_input:
            if args.full_training_gaussian:
                trainingSampleCov = np.cov(trainingImages, rowvar=False)
            else:
                trainingSampleCov = np.cov(trainingData.images, rowvar=False)
            # needed for conditional covariance matrixes, might as well compute just once
            trainingSampleCovInv = np.linalg.pinv(trainingSampleCov)

        # delta method using gradients for parameter uncertainty and moment for input uncertainty
        if not args.skip_delta:
            method = deltaGaussianLogLikelihoodBenchmarkTime if args.benchmark_time else deltaGaussianLogLikelihood
            run_experiment("Moment + Delta", percent, method, psdd, lgc, zero_grad=True)
            if args.parameter_baseline:
                run_experiment("Expectation + Delta", percent, deltaNoInputLogLikelihood, psdd, lgc, zero_grad=True)
                run_experiment("Imputation + Delta", percent, deltaParamLogLikelihood, trainingSampleMean, lgc, zero_grad=True)

        # exact delta should be more accurate than regular delta
        if args.exact_delta:
            run_experiment("Exact Delta", percent, exactDeltaGaussianLogLikelihood, psdd, lgc, zero_grad=True)

        lgc.zero_grad(False)

        enforceBoolean = not args.skip_enforce_boolean
        if args.include_trivial or args.include_residual_input:
            def basicMeanImputation(inputs: np.ndarray):
                return meanImputation(inputs, trainingSampleMean, enforceBoolean=enforceBoolean)

            def basicConditionalImputation(inputs: np.ndarray):
                return conditionalMeanImputation(inputs, trainingSampleMean, trainingSampleCov,
                                                 enforceBoolean=enforceBoolean)

            def basicPsddMpeImputation(inputs: np.ndarray):
                return psddMpeImputation(inputs, psdd)

            # Basic imputation: runs the regressor handling missing values via imputation
            # Expectation is Expected Predictions first moment
            if args.include_trivial:
                run_experiment("RC Mean Imputation", percent, basicImputation, basicMeanImputation, lgc)
                run_experiment("RC Gaussian Imputation", percent, basicImputation, basicConditionalImputation, lgc)
                run_experiment("RC PSDD Imputation", percent, basicImputation, basicPsddMpeImputation, lgc)
                run_experiment("Expectation", percent, basicExpectation, psdd, lgc)
                if nn is not None:
                    run_experiment("NN Mean Imputation", percent, basicNNImputation, basicMeanImputation, nn)
                    run_experiment("NN Gaussian Imputation", percent, basicNNImputation,
                                   basicConditionalImputation, nn)
                    run_experiment("NN PSDD Imputation", percent, basicNNImputation, basicPsddMpeImputation, nn)

            # Residual input runs any of the above basic methods, making like samples with known labels to produce a
            # baseline uncertainty
            if args.include_residual_input:
                run_experiment("RC Imputation + Residual", percent,
                               residualPerSampleInput, pureValidSet,
                               basicImputation, basicMeanImputation, lgc)
                run_experiment("RC Gaussian + Residual", percent,
                               residualPerSampleInput, pureValidSet,
                               basicImputation, basicConditionalImputation, lgc)
                run_experiment("RC PSDD + Residual", percent,
                               residualPerSampleInput, pureValidSet,
                               basicImputation, basicPsddMpeImputation, lgc)
                run_experiment("Expectation + Residual", percent,
                               residualPerSampleInput, pureValidSet,
                               basicExpectation, psdd, lgc)
                if nn is not None:
                    run_experiment("NN Imputation + Residual", percent,
                                   residualPerSampleInput, pureValidSet,
                                   basicNNImputation, basicMeanImputation, nn)
                    run_experiment("NN Gaussian + Residual", percent,
                                   residualPerSampleInput, pureValidSet,
                                   basicNNImputation, basicConditionalImputation, nn)
                    run_experiment("NN PSDD + Residual", percent,
                                   residualPerSampleInput, pureValidSet,
                                   basicNNImputation, basicPsddMpeImputation, nn)

        # Handles input uncertainty simply as the second moment
        if args.input_baseline:
            method = inputLogLikelihoodBenchmarkTime if args.benchmark_time else inputLogLikelihood
            run_experiment("Moment only", percent, method, psdd, lgc)

        # Handle input uncertainty as samples from a monte carlo gaussian
        if args.input_samples > 1:
            run_experiment("RC Gaussian MC {} only".format(args.input_samples), percent,
                           monteCarloGaussianInputOnlyLogLikelihood, lgc,
                           trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                           conditionalGaussian, randState, enforceBoolean)
            run_experiment("RC Marginalize MC {} only".format(args.input_samples), percent,
                           monteCarloGaussianInputOnlyLogLikelihood, lgc,
                           trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                           marginalizeGaussian, randState, enforceBoolean)
            if nn is not None:
                run_experiment("NN Gaussian MC {} only".format(args.input_samples), percent,
                               monteCarloGaussianNNInputOnlyLogLikelihood, nn,
                               trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                               conditionalGaussian, randState, enforceBoolean)
                run_experiment("NN Marginalize MC {} only".format(args.input_samples), percent,
                               monteCarloGaussianNNInputOnlyLogLikelihood, nn,
                               trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                               marginalizeGaussian, randState, enforceBoolean)

        # Handle input uncertainty as samples from a monte carlo psdd
        if args.psdd_samples > 1:
            run_experiment("RC PSDD MC {} only".format(args.psdd_samples), percent,
                           monteCarloPSDDInputOnlyLogLikelihood, psdd, lgc, args.psdd_samples, randState)
            if nn is not None:
                run_experiment("NN PSDD MC {} only".format(args.psdd_samples), percent,
                               monteCarloPSDDNNInputOnlyLogLikelihood, psdd, nn, args.psdd_samples, randState)

        # Monte carlo over parameter uncertainty
        # Fast monte carlo, lets me get the accuracy far closer to Delta with less of a runtime hit
        if args.samples > 1:
            params = sampleMonteCarloParameters(lgc, args.samples, randState)
            method = monteCarloGaussianLogLikelihood if args.benchmark_time else fastMonteCarloGaussianLogLikelihood
            if not args.skip_mc:
                run_experiment("Moment + MC {}".format(args.samples), percent, method, psdd, lgc, params, False)
            if args.parameter_baseline:
                # Alternative parameter baseline, just ignores input uncertainty without mean imputation
                run_experiment("Expectation + MC {}".format(args.samples), percent, method, psdd, lgc, params, True)
                # TODO: consider conditional variant here
                # Standard baseline with mean imputation
                run_experiment("RC Imputation + MC {}".format(args.samples), percent,
                               monteCarloParamLogLikelihood, trainingSampleMean, lgc, params)

            if args.input_samples > 1:
                run_experiment("RC Gaussian MC {} + MC {}".format(args.input_samples, args.samples), percent,
                               monteCarloGaussianParamInputLogLikelihood, lgc, params,
                               trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                               conditionalGaussian, randState, enforceBoolean
                )
                run_experiment("RC Marginalize MC {} + MC {}".format(args.input_samples, args.samples), percent,
                               monteCarloGaussianParamInputLogLikelihood, lgc, params,
                               trainingSampleMean, trainingSampleCov, args.input_samples, trainingSampleCovInv,
                               marginalizeGaussian, randState, enforceBoolean
                )
            if args.psdd_samples > 1:
                run_experiment("PSDD MC {} + MC {}".format(args.psdd_samples, args.samples), percent,
                               monteCarloPSDDParamInputLogLikelihood, psdd, lgc, params, args.psdd_samples, randState
                )

        # BIG WARNING: during the calculations of monte carlo methods, lgc.parameters is the mean while the nodes
        # have their values set to values from the current sample of the parameters. Most other methods assume the
        # parameters are the mean as those tend to perform the best. As a result any non-MC method placed after a MC
        # method will behave poorly

        # We could of course reset the parameters after each trial to the mean value, but it did not seem necessary,
        # sorting the test is simpler and makes the experiments run slightly faster.

        gc.collect()

    resultsSummaryFile.close()
    allResultsFile.close()

    # results
    if args.log_results:
        formatStr = "{:<25} {:<15} {:<15} " \
                    "{:<20} {:<20} " \
                    "{:<25} {:<25} {:<25} {:<25} " \
                    "{:<25} {:<25} {:<25} {:<25} " \
                    "{:<20} {:<20}"
        # this is saved as a CSV, does not need to be in the log
        print(formatStr.format(*csvHeaders))
        print("")
        for result in results:
            # this is saved as a CSV, does not need to be in the log
            print(formatStr.format(*result.getResultRow()))
