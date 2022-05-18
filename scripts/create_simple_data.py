import argparse
import gzip
import logging
import numpy as np
import os
import pickle
import sys

"""Creates a bit field as an array for the given number"""
def bitfield(number: int, size: int):
    return [number >> i & 1 for i in range(size - 1, -1, -1)]

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='main argument')

    parser.add_argument('-o', '--output', type=str,
                        default='./data/',
                        help='Output path for processed data')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--features', type=int,
                        help='Number of features to include')

    parser.add_argument('--train_count', type=int,
                        help='Number of values to include in training')

    parser.add_argument('--min', type=int, default=0,
                        help='Min value to use for a Y value. If excluded, does 0')

    parser.add_argument('--max', type=int, default=1,
                        help='Max value to use for a Y value. If excluded, does boolean (0 or 1)')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    totalSamples = 2**args.features
    if args.train_count > totalSamples:
        raise Exception("Training count must be less than the number of total samples")

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    out_path = args.output + args.dataset + "/"
    os.makedirs(out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    out_log_path = os.path.join(out_path,  'exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

    # generate a list of all features, and a Y value for each feature
    indexes = range(totalSamples)
    X = np.array([bitfield(n, args.features) for n in indexes])
    logging.info(f"X: \n{str(X)}")
    Y = rand_gen.randint(args.min, args.max + 1, size=totalSamples)
    # Y = X[:, 1] + X[:, 2] - X[:, 3] + 2*X[:, 0] + 5
    logging.info(f"Y: {str(Y)}")

    # Determine indexes to save as our training set
    trainIndexes = rand_gen.choice(indexes, size=args.train_count, replace=False)
    logging.info(f"Training: {str(trainIndexes)}")

    # Build final datasets
    x_train = x_valid = X[trainIndexes]
    y_train = y_valid = Y[trainIndexes]
    logging.info(f"X train: \n{str(x_train)}")
    logging.info(f"Y train: {str(y_train)}")
    x_test = X
    y_test = Y
    f_map = np.array(range(args.features))

    #
    # save feature_map
    f_map_path = os.path.join(out_path, f'fmap-{args.dataset}.pickle')
    with open(f_map_path, 'wb') as f:
        pickle.dump(f_map, f)

    #
    # save data, gzipping
    data_output_path = os.path.join(out_path, f'{args.dataset}.pklz')
    with gzip.open(data_output_path, 'wb') as f:
        pickle.dump(((x_train, y_train), (x_valid, y_valid), (x_test, y_test)), f)
    logging.info(f'saved data to {data_output_path}')
