import math
import sys
sys.path.append('.')



import argparse

try:
    from time import perf_counter
except:
    from time import time as perf_counter

import datetime
import os
import logging
import pickle
import gzip
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from LogisticCircuit.util.DataSet import DataSet
from LogisticCircuit.algo.NeuralNetworkBaseline import learn_neural_network


def dump_data_csv(X, data_path):
    with open(data_path, 'w') as f:
        for x in X:
            f.write('{}\n'.format(','.join(str(s) for s in x)))


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help='Path to data dir')
    parser.add_argument('-o', '--output', type=str, default='./nn/', help='Output path to trained network')
    parser.add_argument('--seed', type=int, nargs='?', default=1337, help='Seed for the random generator')
    parser.add_argument('--exp-id', type=str, default=None, help='Dataset output suffix')
    parser.add_argument("--n-iter", type=int, default=200, help='Number of training iterations')

    parser.add_argument("--data_percents", type=float, nargs='*',
                        help="Percentages of the dataset to use in training")
    parser.add_argument("--enforce_subsets", action='store_true',
                        help="If set, each percent subset of parameters will always be a subset of any larger percentage")
    parser.add_argument("--sample_index_folder", type=str, default=None, help='Folder containing train sample indexes')

    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # creating output dirs if they do not exist
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(args.dataset).replace('.pklz', '')

    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))
    os.makedirs(out_path, exist_ok=True)

    #
    # Logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()

    # to file
    log_dir = os.path.join(out_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler("{0}/nn.log".format(log_dir))
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

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    randState = np.random.RandomState(args.seed)

    out_log_path = os.path.join(out_path,  'exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

    #
    # loading up datasets
    with gzip.open(args.dataset, 'rb') as f:
        data_splits = pickle.load(f)

    #
    # unpacking splits
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_splits

    # if not args.regression:
    #     y_train = y_train.astype(np.int8)
    #     y_valid = y_valid.astype(np.int8)
    #     y_test = y_test.astype(np.int8)

    n_features = x_train.shape[1]
    assert x_valid.shape[1] == n_features
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[1] == n_features
    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    logging.info(f'\nLoaded dataset splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    one_hot = False  # not args.regression
    full_train_data = DataSet(x_train, y_train, one_hot)
    valid_data = DataSet(x_valid, y_valid, one_hot)
    test_data = DataSet(x_test, y_test, one_hot)

    logging.info("Starting training...")
    totalSamples = x_train.shape[0]
    indexes = None
    if args.enforce_subsets:
        indexes = randState.permutation(totalSamples)

    # create directory for history, keep clutter down
    hist_dir = os.path.join(out_path, 'history')
    os.makedirs(hist_dir, exist_ok=True)

    for i, percent in enumerate(args.data_percents):
        logging.info("Selecting samples...")

        # load sample indexes we used with the regression circuit
        if args.sample_index_folder is not None:
            with open(args.sample_index_folder + str(percent*100) + "percentSamples.txt", 'r') as sample_file:
                sampleIndexes = [int(v.strip()) for v in sample_file.readlines()]
        else:
            sampleCount = max(2, math.floor(totalSamples * percent))
            if args.enforce_subsets:
                sampleIndexes = indexes[0:sampleCount]
            else:
                sampleIndexes = randState.choice(totalSamples, size=sampleCount, replace=False)

        x = x_train[sampleIndexes, :]
        y = y_train[sampleIndexes]
        train_data = DataSet(x, y, one_hot)

        logging.info(f"Training network for {percent * 100} percent with {y.shape[0]} samples...")
        pl_start_t = perf_counter()

        nn, train_history = learn_neural_network(
            train=train_data,
            valid=valid_data,
            max_iter=args.n_iter,
            seed=randState.randint(0, 2**32 - 1),
        )
        pl_end_t = perf_counter()

        logging.info(f"done learning in {pl_end_t - pl_start_t} secs")
        train_acc = nn.calculate_error(train_data)
        logging.info(f'\ttrain error: {train_acc:.5f}')
        full_train_acc = nn.calculate_error(full_train_data)
        logging.info(f'\tfull train error: {train_acc:.5f}')
        valid_acc = nn.calculate_error(valid_data)
        logging.info(f'\tvalid error: {valid_acc:.5f}')
        test_acc = nn.calculate_error(test_data)
        logging.info(f'\ttest error: {test_acc:.5f}')

        #
        # save circuit
        circuit_path = os.path.join(out_path, f'{str(percent * 100)}percent.pickle')
        with open(circuit_path, 'wb') as f:
            pickle.dump(nn, f)
        logging.info(f'Neural Network saved to {circuit_path}')

        #
        # save training performances
        perf_path = os.path.join(hist_dir, f'{str(percent * 100)}.train-hist')
        np.save(perf_path, train_history)
        logging.info(f'Training history saved to {perf_path}')

        #
        # and plot it
        perf_path = os.path.join(hist_dir, f'{str(percent * 100)}.train-hist.pdf')
        plt.plot(np.arange(len(train_history)), train_history)
        plt.savefig(perf_path)
        plt.close()


