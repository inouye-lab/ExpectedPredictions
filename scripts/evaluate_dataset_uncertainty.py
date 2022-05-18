import math
import sys

import argparse
from typing import Tuple

import gzip
import pickle

import torch
from numpy.random import RandomState
import pypsdd.psdd_io

from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree
from LogisticCircuit.util.DataSet import DataSet

from pypsdd.vtree import Vtree as PSDD_Vtree
from pypsdd.manager import PSddManager

from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, sampleMonteCarloParameters, monteCarloPrediction
from uncertainty_validation import deltaGaussianLogLikelihood, monteCarloGaussianLogLikelihood

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

    parser.add_argument('model', type=str, help='Model to use for expectations')
    parser.add_argument("--classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--samples", type=int, required=True,
                        help="Number of monte carlo samples")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for dataset selection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--missing", type=float, default=0.5,
                        help="Percent of data to treat as missing")
    #
    # parsing the args
    args = parser.parse_args()

    FOLDER = args.model
    VTREE_FILE = FOLDER + ".vtree"
    GLC_FILE = FOLDER + ".glc"
    PSDD_FILE = FOLDER + ".psdd"

    lc_vtree = LC_Vtree.read(VTREE_FILE)
    with open(GLC_FILE) as circuit_file:
        lgc = LogisticCircuit(lc_vtree, args.classes, circuit_file=circuit_file, requires_grad=True)

    print("Loading samples...")
    with gzip.open(args.data, 'rb') as f:
        rawData = pickle.load(f)
    train, test, valid = rawData
    testSet = _toDataset(test)

    print("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    # TODO: missing values
    randState = RandomState(args.seed)
    variables = testSet.images.shape[1]
    sampleIndexes = randState.choice(variables, size=math.floor(variables * args.missing), replace=False)
    testSet.images[:, sampleIndexes] = -1  # internal value representing missing

    # delta method
    start_t = perf_counter()
    dTotalError, diLikelihood, dpLikelihood, dtLikelihood, \
        diVariance, dpVariance, dtVariance = deltaGaussianLogLikelihood(psdd, lgc, testSet)
    end_t = perf_counter()
    print("Delta method took {}".format(end_t - start_t))

    # monte carlo
    lgc.zero_grad(False)
    params = sampleMonteCarloParameters(lgc, args.samples, randState)
    mcTotalError, mciLikelihood, mcpLikelihood, mctLikelihood, \
        mciVariance, mcpVariance, mctVariance = monteCarloGaussianLogLikelihood(psdd, lgc, testSet, params)

    print(f"Finished all monte carlo predictions")
    end_t = perf_counter()
    print("Monte carlo took {}".format(end_t - start_t))

    # results
    print("{:<15} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25}"
          .format("Name", "Total Error", "Input LL", "Param LL", "Total LL", "Input Var", "Param Var", "Total Var"))
    print("")
    print("{:<15} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25}"
          .format("Monte Carlo", mcTotalError.item(), mciLikelihood.item(), mcpLikelihood.item(), mctLikelihood.item(),
                  mciVariance.item(), mcpVariance.item(), mctVariance.item()))
    print("{:<15} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25} {:<25}"
          .format("Delta Method", dTotalError.item(), diLikelihood.item(), dpLikelihood.item(), dtLikelihood.item(),
                  diVariance.item(), dpVariance.item(), dtVariance.item()))
