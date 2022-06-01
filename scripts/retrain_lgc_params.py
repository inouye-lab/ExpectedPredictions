import argparse

import gzip
import math

import pickle
from datetime import datetime

import numpy as np
import os
import torch
from numpy.random import RandomState
from typing import Tuple, List

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

    # Load in existing model
    modelBase = args.prefix + args.model
    vtreeFile = modelBase + ".vtree"
    glcFle = modelBase + ".glc"

    print("Loading circuit...")
    vtree = LC_Vtree.read(vtreeFile)
    randState = RandomState(args.seed)
    with open(glcFle) as circuit_file:
        if args.regression:
            lgc = RegressionCircuit(vtree, circuit_file=circuit_file, rand_gen=randState)
        else:
            lgc = LogisticCircuit(vtree, args.classes, circuit_file=circuit_file, rand_gen=randState)

    print("Loading samples...")
    with gzip.open(args.data, 'rb') as f:
        rawData = pickle.load(f)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = rawData
    if not args.regression:
        y_train = y_train.astype(np.int8)
        y_valid = y_valid.astype(np.int8)
        y_test = y_test.astype(np.int8)

    print(f'\nLoaded dataset splits of shapes:')
    print(f'\t\ttrain {x_train.shape} {y_train.shape}')
    print(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    print(f'\t\ttest  {x_test.shape} {y_test.shape}')

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

    circuitRoot = args.prefix + args.output
    if args.exp_id:
        circuitRoot = os.path.join(circuitRoot, args.exp_id)
    else:
        circuitRoot = os.path.join(circuitRoot, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(circuitRoot, exist_ok=True)

    if not args.keep_params:
        print("Resetting parameters...")
        lgc.randomize_node_parameters()

    print("Saving original parameters...")
    originalParams = lgc.parameters.clone()

    # train for each selected percent
    totalSamples = x_combined.shape[0]
    indexes = None
    if args.enforce_subsets:
        indexes = randState.permutation(totalSamples)
    for i, percent in enumerate(args.data_percents):
        print("\nRestoring parameters...")
        lgc.set_node_parameters(originalParams.clone(), set_circuit=True, reset_covariance=True)

        print("Selecting samples...")
        sampleCount = max(2, math.floor(totalSamples * percent))
        if args.enforce_subsets:
            sampleIndexes = indexes[0:sampleCount]
        else:
            sampleIndexes = randState.choice(totalSamples, size=sampleCount, replace=False)
        x = x_combined[sampleIndexes, :]
        y = y_combined[sampleIndexes]
        train_data = DataSet(x, y, one_hot)
        train_data.features = lgc.calculate_features(train_data.images)

        print(f"Training circuit for {percent * 100} percent with {y.shape[0]} samples...")
        if args.regression:
            pl_start_t = perf_counter()
            lgc.learn_parameters(
                train_data, args.n_iter_pl, rand_gen=randState, solver=args.solver,
                params={
                    'scoreLL': True,
                    # 'lambda_init': 0.1
                },
                cv_params={
                    'lambda_init': [0.01, 0.1, 1, 10, 100],
                }
            )
            pl_end_t = perf_counter()

            train_acc: float = lgc.calculate_error(train_data)
            full_train_acc: float = lgc.calculate_error(full_train_data)
            valid_err: float = lgc.calculate_error(valid_data)
            test_err: float = lgc.calculate_error(test_data)

            print(f"done learning in {pl_end_t - pl_start_t} secs")
            print(f"\tcircuit size: {lgc.num_parameters}")
            print(f"\terror train: {train_acc:.5f}")
            print(f"\terror full train: {full_train_acc:.5f}")
            print(f"\terror test: {valid_err:.5f}")
            print(f"\terror valid: {test_err:.5f}")

        else:
            pl_start_t = perf_counter()
            lgc.learn_parameters(train_data, args.n_iter_pl, rand_gen=randState, solver=args.solver)
            pl_end_t = perf_counter()

            train_acc: float = lgc.calculate_accuracy(train_data)
            full_train_acc: float = lgc.calculate_accuracy(full_train_data)
            valid_err = lgc.calculate_accuracy(valid_data)
            test_err = lgc.calculate_accuracy(test_data)

            print(f"done learning in {pl_end_t - pl_start_t} secs")
            print(f"\tcircuit size: {lgc.num_parameters}")
            print(f"\taccuracy train: {train_acc:.5f}")
            print(f"\taccuracy full train: {full_train_acc:.5f}")
            print(f"\taccuracy test: {valid_err:.5f}")
            print(f"\taccuracy valid: {test_err:.5f}")

        # save circuit
        circuitPath = circuitRoot + '/' + str(percent * 100) + 'percent.glc'
        with open(circuitPath, 'w') as f:
            lgc.save(f)
        print(f'Circuit saved to {circuitPath}')
        samplePath = circuitRoot + '/' + str(percent * 100) + 'percentSamples.txt'
        with open(samplePath, 'w') as f:
            for i in sampleIndexes:
                f.write(str(i) + '\n')
