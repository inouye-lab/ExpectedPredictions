import argparse
import json
import gzip
import math

import pickle
import logging
from datetime import datetime

import numpy as np
import os
import torch
from numpy.random import RandomState
from typing import Tuple, List
import sys

sys.path.append('.')

from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.algo.RegressionCircuit import RegressionCircuit
from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree
from LogisticCircuit.util.DataSet import DataSets, DataSet

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def _toDataset(data: Tuple[torch.Tensor, torch.Tensor]) -> DataSet:
    """Convers the given tuple into a dataset object"""
    return DataSet(data[0], data[1], one_hot = False)


if __name__ == '__main__':
    print("Loading Logistic Circuit..")
    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='Original model to be retrained, added to the prefix')
    parser.add_argument('--prefix', type=str, default='',
                        help='Folder prefix for both the model and the output')
    parser.add_argument('--output', type=str,  required=True,
                        help='Output path, added to the prefix')

    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for dataset selection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset")

    parser.add_argument("--regression",  action='store_true',
                        help="Regression instead of classification")
    parser.add_argument("--solver", type=str, default="auto",
                        help="Method used to compute the parameters for the circuit")
    parser.add_argument('--params', type=json.loads, default=None, help='Additional arguments to pass to the solver')
    parser.add_argument('--cv_params', type=json.loads, default=None, help='If set, additional solver arguments to perform cross validation over')

    parser.add_argument("--keep_params",  action='store_true',
                        help="If set, does not discard the parameters of the circuit before training. By default the original parameters are discarded")
    parser.add_argument("--use_valid",  action='store_true',
                        help="If set, merges the validation data into the training data for parameters")
    parser.add_argument("--enforce_subsets",  action='store_true',
                        help="If set, each percent subset of parameters will always be a subset of any larger percentage")
    parser.add_argument("--n-iter-pl", type=int, default=15,
                        help="Number of iterations of parameter learning to run")

    parser.add_argument("--data_percents", type=float, nargs='*',
                        help="Percentages of the dataset to use in training")

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    #
    # parsing the args
    args = parser.parse_args()

    # determine output folder
    circuitRoot = args.prefix + args.output
    if args.exp_id:
        circuitRoot = os.path.join(circuitRoot, args.exp_id)
    else:
        circuitRoot = os.path.join(circuitRoot, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(circuitRoot, exist_ok=True)

    # setup logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()

    # to file
    log_dir = os.path.join(args.output, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler("{0}/retrain.log".format(circuitRoot))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # and to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # setting verbosity level
    if args.verbose == 1:
        root_logger.setLevel(logging.INFO)
    elif args.verbose == 2:
        root_logger.setLevel(logging.DEBUG)

    # Print welcome message
    args_out_path = os.path.join(circuitRoot, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    # Load in existing model
    modelBase = args.prefix + args.model
    vtreeFile = modelBase + ".vtree"
    glcFle = modelBase + ".glc"

    logging.info("Loading circuit...")
    vtree = LC_Vtree.read(vtreeFile)
    randState = RandomState(args.seed)
    with open(glcFle) as circuit_file:
        if args.regression:
            lgc = RegressionCircuit(vtree, circuit_file=circuit_file, rand_gen=randState)
        else:
            lgc = LogisticCircuit(vtree, args.classes, circuit_file=circuit_file, rand_gen=randState)

    logging.info("Loading samples...")
    with gzip.open(args.data, 'rb') as f:
        rawData = pickle.load(f)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = rawData
    if not args.regression:
        y_train = y_train.astype(np.int8)
        y_valid = y_valid.astype(np.int8)
        y_test = y_test.astype(np.int8)

    logging.info(f'\nLoaded dataset splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    # create baseline datasets
    one_hot = not args.regression
    full_train_data = DataSet(x_train, y_train, one_hot)
    valid_data = DataSet(x_valid, y_valid, one_hot)
    test_data = DataSet(x_test, y_test, one_hot)

    combined_data = full_train_data
    x_combined = x_train
    y_combined = y_train
    if args.use_valid:
        x_combined = np.concatenate((x_train, x_valid), axis=0)
        y_combined = np.concatenate((y_train, y_valid), axis=0)

    full_train_data.features = lgc.calculate_features(full_train_data.images)
    valid_data.features = lgc.calculate_features(valid_data.images)
    test_data.features = lgc.calculate_features(test_data.images)

    # select random data for training

    # reset parameters unless told to not
    if not args.keep_params:
        logging.info("Resetting parameters...")
        lgc.randomize_node_parameters()

    logging.info("Saving original parameters...")
    originalParams = lgc.parameters.clone()

    # train for each selected percent
    totalSamples = x_combined.shape[0]
    indexes = None
    if args.enforce_subsets:
        indexes = randState.permutation(totalSamples)
    eps = np.finfo(np.float64).eps
    for i, percent in enumerate(args.data_percents):
        logging.info("\nRestoring parameters...")
        lgc.set_node_parameters(originalParams.clone(), set_circuit=True, reset_covariance=True)

        logging.info("Selecting samples...")
        sampleCount = max(2, math.floor(totalSamples * percent))
        if args.enforce_subsets:
            sampleIndexes = indexes[0:sampleCount]
        else:
            sampleIndexes = randState.choice(totalSamples, size=sampleCount, replace=False)
        x = x_combined[sampleIndexes, :]
        y = y_combined[sampleIndexes]
        train_data = DataSet(x, y, one_hot)
        train_data.features = lgc.calculate_features(train_data.images)

        logging.info(f"Training circuit for {percent * 100} percent with {y.shape[0]} samples...")
        if args.regression:
            pl_start_t = perf_counter()
            cv_params = args.cv_params
            if cv_params is not None and "alpha_init_scale" in cv_params:
                cv_params = cv_params.copy()
                var = 1. / (np.var(train_data.labels.numpy()) + eps)
                cv_params["alpha_init"] = [x * var for x in cv_params["alpha_init_scale"]]
                cv_params.pop('alpha_init_scale', None)
                logging.info(f"Using scaled alpha_init {cv_params['alpha_init']}")

            lgc.learn_parameters(
                train_data, args.n_iter_pl, rand_gen=randState, solver=args.solver,
                params=args.params, cv_params=cv_params
            )
            pl_end_t = perf_counter()

            train_acc: float = lgc.calculate_error(train_data)
            full_train_acc: float = lgc.calculate_error(full_train_data)
            valid_err: float = lgc.calculate_error(valid_data)
            test_err: float = lgc.calculate_error(test_data)

            logging.info(f"done learning in {pl_end_t - pl_start_t} secs")
            logging.info(f"\tcircuit size: {lgc.num_parameters}")
            logging.info(f"\terror train: {train_acc}")
            logging.info(f"\terror full train: {full_train_acc}")
            logging.info(f"\terror test: {valid_err}")
            logging.info(f"\terror valid: {test_err}")

        else:
            pl_start_t = perf_counter()
            lgc.learn_parameters(train_data, args.n_iter_pl, rand_gen=randState, solver=args.solver,
                                 params=args.params, cv_params=args.cv_params)
            pl_end_t = perf_counter()

            train_acc: float = lgc.calculate_accuracy(train_data)
            full_train_acc: float = lgc.calculate_accuracy(full_train_data)
            valid_err = lgc.calculate_accuracy(valid_data)
            test_err = lgc.calculate_accuracy(test_data)

            logging.info(f"done learning in {pl_end_t - pl_start_t} secs")
            logging.info(f"\tcircuit size: {lgc.num_parameters}")
            logging.info(f"\taccuracy train: {train_acc}")
            logging.info(f"\taccuracy full train: {full_train_acc}")
            logging.info(f"\taccuracy test: {valid_err}")
            logging.info(f"\taccuracy valid: {test_err}")

        # save circuit
        circuitPath = circuitRoot + '/' + str(percent * 100) + 'percent.glc'
        with open(circuitPath, 'w') as f:
            lgc.save(f)
        logging.info(f'Circuit saved to {circuitPath}')
        samplePath = circuitRoot + '/' + str(percent * 100) + 'percentSamples.txt'
        with open(samplePath, 'w') as f:
            for i in sampleIndexes:
                f.write(str(i) + '\n')
