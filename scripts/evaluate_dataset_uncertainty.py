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

from pypsdd.vtree import Vtree as PSDD_Vtree
from pypsdd.manager import PSddManager

from uncertainty_calculations import sampleMonteCarloParameters
from uncertainty_validation import deltaGaussianLogLikelihood, monteCarloGaussianLogLikelihood, \
    fastMonteCarloGaussianLogLikelihood, exactDeltaGaussianLogLikelihood, monteCarloParamLogLikelihood, \
    deltaParamLogLikelihood, inputLogLikelihood, SummaryType, deltaGaussianLogLikelihoodBenchmarkTime, \
    inputLogLikelihoodBenchmarkTime

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
    totalError: Tensor

    inputLL: Tensor
    paramLL: Tensor
    totalLL: Tensor

    inputVar: Tensor
    paramVar: Tensor
    totalVar: Tensor

    def __init__(self, method: str, trainPercent: float, missingPercent: float,
                 totalError: Tensor,
                 inputLL: Tensor, paramLL: Tensor, totalLL: Tensor,
                 inputVar: Tensor, paramVar: Tensor, totalVar: Tensor):
        self.method = method
        self.trainPercent = trainPercent
        self.missingPercent = missingPercent
        self.totalError = totalError

        self.inputLL = inputLL
        self.paramLL = paramLL
        self.totalLL = totalLL

        self.inputVar = inputVar
        self.paramVar = paramVar
        self.totalVar = totalVar

        self.runtime = None

    def print(self):
        logging.info(f"{self.method} @ train {self.trainPercent}, missing {self.missingPercent}")
        logging.info(f"    error: {self.totalError.item()}")
        logging.info(f"    ll: input {self.inputLL.item()}, param {self.paramLL.item()}, total {self.totalLL.item()}")
        logging.info(f"    var: input {self.inputVar.item()}, param {self.paramVar.item()}, total {self.totalVar.item()}")

    def getResultRow(self):
        return [
            self.method, self.trainPercent, self.missingPercent,
            self.runtime, self.totalError.item(),
            self.inputLL.item(), self.paramLL.item(), self.totalLL.item(),
            self.inputVar.item(), self.paramVar.item(), self.totalVar.item()
        ]


if __name__ == '__main__':
    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='Model to use for expectations')
    parser.add_argument('--prefix', type=str, default='',
                        help='Folder prefix for both the model and the output')
    parser.add_argument('--output', type=str, help='Location for result csv')
    parser.add_argument("--classes", type=int, required=True,
                        help="Number of classes in the dataset")

    parser.add_argument("--skip_delta",  action='store_true',
                        help="If set, the delta method is skipped, running just MC")
    parser.add_argument("--exact_delta",  action='store_true',
                        help="If set, runs the exact delta method")
    parser.add_argument("--parameter_baseline",  action='store_true',
                        help="If set, runs the baseline parameter uncertainty using the dataset mean")
    parser.add_argument("--input_baseline",  action='store_true',
                        help="If set, runs the baseline input uncertainty using the parameter mean")
    parser.add_argument("--global_missing_features",  action='store_true',
                        help="If set, the same feature will be missing in all samples. If unset, each sample will have missing features selected separately")
    parser.add_argument("--samples", type=int, default=0,
                        help="Number of monte carlo samples")
    parser.add_argument("--benchmark_time",  action='store_true',
                        help="If set, disables batching on several methods to make the times more comparable")
    parser.add_argument("--evaluate_validation",  action='store_true',
                        help="If set, evaluates the validation dataset instead of the testing dataset. Used to validate against known data.")

    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for dataset selection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--missing", type=float, nargs='*',
                        help="Percent of data to treat as missing")
    parser.add_argument("--retrain_dir", type=str, required=True,
                        help="Location of folders for retrained models")
    parser.add_argument("--data_percents", type=float, nargs='*',
                        help="Percentages of the dataset to use in training")

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
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
    # TODO: that might be the wrong dataset for evaluation
    (trainingImages, trainingLabels), (validImages, validLabels), (images, labels) = rawData
    if args.evaluate_validation:
        images = validImages
        labels = validLabels

    logging.info("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    # populate missing datasets
    logging.info("Preparing missing datasets")
    randState = RandomState(args.seed)
    testSets: List[Tuple[float, DataSet]] = []
    samples = images.shape[0]
    variables = images.shape[1]
    for missing in args.missing:
        testImages = np.copy(images)
        if args.global_missing_features:
            sampleIndexes = randState.choice(variables, size=math.floor(variables * missing), replace=False)
            testImages[:, sampleIndexes] = -1  # internal value representing missing
        else:
            for i in range(samples):
                sampleIndexes = randState.choice(variables, size=math.floor(variables * missing), replace=False)
                testImages[i, sampleIndexes] = -1  # internal value representing missing
        testSets.append((missing, DataSet(testImages, labels, one_hot = False)))

    # first loop is over percents
    results: List[Result] = []

    # start CSV files for output
    resultsSummaryFile = open(os.path.join(args.output, outputName + "-summary.csv"), 'w')
    allResultsFile = open(os.path.join(args.output, outputName + "-all.csv"), 'w')
    resultsSummary = csv.writer(resultsSummaryFile)
    allResults = csv.writer(allResultsFile)
    csvHeaders = [
        "Name", "Train Percent", "Missing Percent",
        "Runtime", "Total Error",
        "Input LL", "Param LL", "Total LL",
        "Input Var", "Param Var", "Total Var"
    ]
    resultsSummary.writerow(csvHeaders)
    allResults.writerow([
        "Name", "Train Percent", "Missing Percent", "Sample Index",
        "Expected", "Mean", "Input Variance", "Parameter Variance"
    ])
    resultsSummaryFile.flush()
    allResultsFile.flush()

    # all experiments follow a general form, so abstract that out a bit
    # Argument signature: experiment_function(*experiment_arguments, testSet)
    def run_experiment(name: str, trainPercent: float, experiment_function, *experiment_arguments,
                       zero_grad: bool = False):
        for (missing, testSet) in testSets:
            testSet: DataSet
            if zero_grad:
                lgc.zero_grad(True)

            # print experiment header
            logging.info("{}: Running {} at {}% training, {}% missing".format(args.model, name, trainPercent * 100, missing * 100))

            # run experiment
            start_t = perf_counter()
            experiment_result: SummaryType = experiment_function(*experiment_arguments, testSet)
            result = Result(name, trainPercent, missing, *experiment_result[0:7])
            end_t = perf_counter()
            result.runtime = end_t - start_t
            results.append(result)

            # save to CSV file
            resultsSummary.writerow(result.getResultRow())
            resultsSummaryFile.flush()
            # save full results
            mean, inputVariance, parameterVariance = experiment_result[7:10]
            length = len(testSet.labels)
            assert len(mean) == length
            assert len(inputVariance) == length
            assert len(parameterVariance) == length
            for sample in range(length):
                allResults.writerow([
                    name, trainPercent, missing, sample,
                    testSet.labels[sample].item(),
                    mean[sample].item(),
                    inputVariance[sample].item(),
                    parameterVariance[sample].item(),
                ])
            allResultsFile.flush()

            # print progress
            result.print()
            logging.info("{}: {} at {}% training and {}% missing took {}"
                         .format(args.model, name, percent * 100, missing * 100, result.runtime))
            logging.info("----------------------------------------------------------------------------------------")

    # main experiment loop
    retrainFolder = args.prefix + args.retrain_dir
    for percent in args.data_percents:
        logging.info("Running {} percent".format(percent*100))
        logging.info("========================================================================================")
        percentFolder = retrainFolder + str(percent*100)
        with open(percentFolder + "percent.glc", 'r') as circuit_file:
            requireGrad = not args.skip_delta or args.exact_delta
            lgc = LogisticCircuit(lc_vtree, args.classes, circuit_file=circuit_file, requires_grad=requireGrad)
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
        if args.parameter_baseline:
            trainingSampleMean = np.mean(trainingData.images, axis=0)

        # second loop is over missing value counts
        if not args.skip_delta:
            method = deltaGaussianLogLikelihoodBenchmarkTime if args.benchmark_time else deltaGaussianLogLikelihood
            run_experiment("Delta Method", percent, method, psdd, lgc, zero_grad=True)

        # second loop is over missing value counts
        if not args.skip_delta and args.parameter_baseline:
            run_experiment("BL Delta Param", percent, deltaParamLogLikelihood, trainingSampleMean, lgc, zero_grad=True)

        # exact delta should be more accurate than regular delta
        if args.exact_delta:
            run_experiment("Exact Delta", percent, exactDeltaGaussianLogLikelihood, psdd, lgc, zero_grad=True)

        lgc.zero_grad(False)

        if args.input_baseline:
            method = inputLogLikelihoodBenchmarkTime if args.benchmark_time else inputLogLikelihood
            run_experiment("BL Input", percent, method, psdd, lgc)

        # Fast monte carlo, lets me get the accuracy far closer to Delta with less of a runtime hit
        if args.samples > 1:
            params = sampleMonteCarloParameters(lgc, args.samples, randState)
            method = monteCarloGaussianLogLikelihood if args.benchmark_time else fastMonteCarloGaussianLogLikelihood
            run_experiment("Monte Carlo", percent, method, psdd, lgc, params)

        # BIG WARNING: during the calculations of monte carlo methods, lgc.parameters is the mean while the nodes
        # have their values set to values from the current sample of the parameters. Most other methods assume the
        # parameters are the mean as those tend to perform the best. As a result any non-MC method placed after a MC
        # method will behave poorly

        # We could of course reset the parameters after each trial to the mean value, but it did not seem necessary,
        # sorting the test is simpler and makes the experiments run slightly faster.

        if args.parameter_baseline and args.samples > 0:
            params = sampleMonteCarloParameters(lgc, args.samples, randState)
            run_experiment("BL MC Param", percent, monteCarloParamLogLikelihood, trainingSampleMean, lgc, params)

        gc.collect()

    resultsSummaryFile.close()
    allResultsFile.close()

    # results
    formatStr = "{:<20} {:<15} {:<15} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}"
    # this is saved as a CSV, does not need to be in the log
    print(formatStr.format(*csvHeaders))
    print("")
    for result in results:
        # this is saved as a CSV, does not need to be in the log
        print(formatStr.format(*result.getResultRow()))
