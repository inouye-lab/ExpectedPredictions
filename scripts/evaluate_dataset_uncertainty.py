import json
import math
import sys

import csv
import gc

import os

import logging
from datetime import datetime

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
    fastMonteCarloGaussianLogLikelihood

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
    totalError: np.ndarray

    inputLL: np.ndarray
    paramLL: np.ndarray
    totalLL: np.ndarray

    inputVar: np.ndarray
    paramVar: np.ndarray
    totalVar: np.ndarray

    def __init__(self, method: str, trainPercent: float, missingPercent: float,
                 totalError: np.ndarray,
                 inputLL: np.ndarray, paramLL: np.ndarray, totalLL: np.ndarray,
                 inputVar: np.ndarray, paramVar: np.ndarray, totalVar: np.ndarray):
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
        print(f"{self.method} @ train {self.trainPercent}, missing {self.missingPercent}")
        print(f"    error: {self.totalError.item()}")
        print(f"    ll: input {self.inputLL.item()}, param {self.paramLL.item()}, total {self.totalLL.item()}")
        print(f"    var: input {self.inputVar.item()}, param {self.paramVar.item()}, total {self.totalVar.item()}")


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
    parser.add_argument("--global_missing_features",  action='store_true',
                        help="If set, the same feature will be missing in all samples. If unset, each sample will have missing features selected separately")
    parser.add_argument("--samples", type=int, default=0,
                        help="Number of monte carlo samples")
    parser.add_argument("--fast_samples", type=int, default=0,
                        help="Number of fast monte carlo samples")

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
    out_folder = os.path.dirname(args.output)
    log_dir = os.path.join(out_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    date_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_dir, date_string))
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
    args_out_path = os.path.join(log_dir, date_string + '.json')
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
    _, (images, labels), _ = rawData

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
                testImages[i, sampleIndexes] = -1 # internal value representing missing
        testSets.append((missing, DataSet(testImages, labels, one_hot = False)))

    # first loop is over percents
    results: List[Result] = []
    for percent in args.data_percents:
        logging.info("Running {} percent".format(percent*100))
        percentFolder = args.prefix + args.retrain_dir
        with open(percentFolder + str(percent*100) + "percent.glc", 'r') as circuit_file:
            lgc = LogisticCircuit(lc_vtree, args.classes, circuit_file=circuit_file, requires_grad=not args.skip_delta)

        # second loop is over missing value counts
        if not args.skip_delta:
            for (missing, testSet) in testSets:
                logging.info("Running {} at {}% missing for delta".format(args.model, missing*100))
                # delta method
                start_t = perf_counter()
                lgc.zero_grad(True)
                result = Result(
                    "Delta Method", percent, missing,
                    *deltaGaussianLogLikelihood(psdd, lgc, testSet)
                )
                result.print()
                end_t = perf_counter()
                result.runtime = end_t - start_t
                results.append(result)
                logging.info("Delta method for {} at {}% training and {}% missing took {}"
                             .format(args.model, percent*100, missing*100, result.runtime))

        # Fast monte carlo, should let me get the accuracy far closer to Delta with less of a runtime hit
        lgc.zero_grad(False)
        if args.fast_samples > 1:
            for (missing, testSet) in testSets:
                logging.info("Running {} at {}% missing for fast monte carlo".format(args.model, missing*100))
                start_t = perf_counter()
                params = sampleMonteCarloParameters(lgc, args.fast_samples, randState)
                result = Result(
                    "Fast MC", percent, missing,
                    *fastMonteCarloGaussianLogLikelihood(psdd, lgc, testSet, params)
                )
                result.print()
                end_t = perf_counter()
                result.runtime = end_t - start_t
                results.append(result)

                logging.info("Fast monte carlo for {} at {}% training and {}% missing took {}"
                             .format(args.model, percent*100, missing*100, result.runtime))

        # monte carlo
        if args.samples > 1:
            for (missing, testSet) in testSets:
                logging.info("Running {} at {}% missing for monte carlo".format(args.model, missing*100))
                start_t = perf_counter()
                params = sampleMonteCarloParameters(lgc, args.samples, randState)
                result = Result(
                    "Monte Carlo", percent, missing,
                    *monteCarloGaussianLogLikelihood(psdd, lgc, testSet, params)
                )
                result.print()
                end_t = perf_counter()
                result.runtime = end_t - start_t
                results.append(result)

                logging.info("Monte carlo for {} at {}% training and {}% missing took {}"
                             .format(args.model, percent*100, missing*100, result.runtime))

        gc.collect()

    # results
    formatStr = "{:<15} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25}"
    headers = [
        "Name", "Train Percent", "Missing Percent",
        "Runtime", "Total Error",
        "Input LL", "Param LL", "Total LL",
        "Input Var", "Param Var", "Total Var"
    ]
    logging.info(formatStr.format(*headers))
    logging.info("")
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for result in results:
            resultRow = [
                result.method, result.trainPercent, result.missingPercent,
                result.runtime, result.totalError.item(),
                result.inputLL.item(), result.paramLL.item(), result.totalLL.item(),
                result.inputVar.item(), result.paramVar.item(), result.totalVar.item()
            ]
            logging.info(formatStr.format(*resultRow))
            writer.writerow(resultRow)
