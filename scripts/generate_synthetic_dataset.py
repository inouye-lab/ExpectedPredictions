import argparse
import json

import logging

import os
import sys
from datetime import datetime

import gzip
import numpy as np
import pickle
from numpy.random import RandomState
import shutil

sys.path.append('.')

import pypsdd.psdd_io
from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.algo.RegressionCircuit import RegressionCircuit
from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree
from pypsdd import PSddManager
from pypsdd.vtree import Vtree as PSDD_Vtree

if __name__ == '__main__':
    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Name of the model, used for output name and default model location')
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder containing the model, default if specific paths are unset')
    parser.add_argument('--vtree', type=str, default=None, help='Path to the vtree for the circuits')
    parser.add_argument('--psdd', type=str, default=None, help='Path to the PSDD parameters')
    parser.add_argument('--gaussian_data', type=str, default=None,
                        help='Path to dataset for gaussian modeling. If set used instead of the PSDD.')
    parser.add_argument('--glc', type=str, default=None, help='Path to the Logistic Circuit parameters')
    parser.add_argument("--classes", type=int, required=True,
                        help="Number of classes in the dataset, if 0 will use regression")

    # Output configuration
    parser.add_argument("--seed", type=int, default=1337, help="Seed for dataset generation")
    parser.add_argument('--output', type=str, required=True, help='Folder for output dataset and logs')
    parser.add_argument('--output_id', type=str, default=None, help='Unique ID for output, if unset uses the date')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Verbosity level')

    # Output size
    parser.add_argument("--train_count", type=int, required=True,
                        help="Number of samples to generate for training")
    parser.add_argument("--valid_count", type=int, required=True,
                        help="Number of samples to generate for validation")
    parser.add_argument("--test_count", type=int, required=True,
                        help="Number of samples to generate for testing")
    parser.add_argument("--fmap_path", type=str, default=None,
                        help="Path to the fmap to clone")

    #
    # parsing the args
    args = parser.parse_args()

    # setup logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()

    # to file
    date_string = args.output_id if args.output_id is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
    outputDir = os.path.join(args.output, args.model, date_string)
    os.makedirs(outputDir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(outputDir, args.model + ".log"))
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

    #
    # find input circuits that we will sample
    def defaultPath(path, extension):
        if path is not None:
            return path
        if args.folder is None:
            raise Exception(f"Must set either {extension} or folder")
        return args.folder + args.model + "." + extension
    vtreePath = defaultPath(args.vtree, "vtree")
    psddPath = defaultPath(args.psdd, "psdd")
    glcPath = defaultPath(args.glc, "glc")

    # Print welcome message
    argsOutPath = os.path.join(outputDir, 'args.json')
    jsonArgs = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", jsonArgs, argsOutPath)
    with open(argsOutPath, 'w') as f:
        f.write(jsonArgs)

    #
    # Load in models
    randState = RandomState(args.seed)
    logging.info("Loading LC..")
    lc_vtree = LC_Vtree.read(vtreePath)
    with open(glcPath, 'r') as circuitFile:
        if args.classes == 0:
            lgc = RegressionCircuit(lc_vtree, circuit_file=circuitFile, rand_gen=randState)
        else:
            lgc = LogisticCircuit(lc_vtree, args.classes, circuit_file=circuitFile, rand_gen=randState)

    #
    # Prepare space for all the generated samples
    totalSamples = args.train_count + args.valid_count + args.test_count

    # When gaussian data is set, sample from a gaussian
    if args.gaussian_data is not None:
        logging.info("Loading samples...")
        with gzip.open(args.gaussian_data, 'rb') as f:
            rawData = pickle.load(f)
        (trainingImages, trainingLabels), (validImages, validLabels), (testImages, testLabels) = rawData
        trainingSampleMean = np.mean(trainingImages, axis=0)
        trainingSampleCov = np.cov(trainingImages, rowvar=False)

        logging.info("Sampling Gaussian..")
        xSamples = np.clip(randState.multivariate_normal(trainingSampleMean, trainingSampleCov, size=totalSamples), 0, 1)
    else:
        # when no gaussian data, sample from the PSDD
        logging.info("Loading PSDD..")
        psdd_vtree = PSDD_Vtree.read(vtreePath)
        manager = PSddManager(psdd_vtree)
        psdd = pypsdd.psdd_io.psdd_yitao_read(psddPath, manager)

        logging.info("Sampling PSDD..")
        xSamples = np.zeros((totalSamples, lgc.num_variables))
        for sample in range(totalSamples):
            inst = psdd.simulate(seed = randState.randint(0, 2**32 - 1))
            for var, value in inst:
                if value == 1:
                    # InstMap is 1 indexed, numpy is zero indexed
                    xSamples[sample, var - 1] = 1

    # generate labels
    features = lgc.calculate_features(xSamples)
    ySamples = lgc.predict(features).numpy().squeeze()

    #
    # save output file
    dataOutputPath = os.path.join(outputDir, args.model + ".pklz")
    with gzip.open(dataOutputPath, 'wb') as f:
        data = (
            (xSamples[0:args.train_count],
             ySamples[0:args.train_count]),
            (xSamples[args.train_count:args.valid_count+args.train_count],
             ySamples[args.train_count:args.valid_count+args.train_count]),
            (xSamples[args.valid_count+args.train_count:totalSamples],
             ySamples[args.valid_count+args.train_count:totalSamples]))
        assert len(data[0][1]) + len(data[1][1]) + len(data[2][1]) == totalSamples
        pickle.dump(data, f)
    logging.info(f'Saved data to {dataOutputPath}')
    # clone fmap
    if args.fmap_path is not None:
        fmapPath = os.path.join(outputDir, f"fmap-{args.model}.pickle")
        shutil.copy(args.fmap_path, fmapPath)
        logging.info(f'Cloned fmap from {args.fmap_path} to {fmapPath}')
