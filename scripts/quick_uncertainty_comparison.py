import sys

import argparse
from typing import List, Tuple

import gzip
import numpy as np
import pickle
from numpy.random import RandomState
from torch import Tensor
import pypsdd.psdd_io
from EVCache import EVCache

from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree

from pypsdd.vtree import Vtree as PSDD_Vtree
from pypsdd.manager import PSddManager

from uncertainty_calculations import deltaMeanAndParameterVariance, deltaInputVariance, sampleMonteCarloParameters, monteCarloPrediction

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def bitfield(number: int, size: int):
    """Creates a bit field as an array for the given number"""
    return [number >> i & 1 for i in range(size - 1, -1, -1)]


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
    parser.add_argument("--count", type=int, default=1,
                        help="Number of values from the dataset to evaluate")
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
        data = pickle.load(f)

    print("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    # Sample X and Y values from the testing set
    randState = RandomState(args.seed)
    sampleIndexes = randState.choice(data[2][1].size, size=args.count, replace=False)
    X = data[2][0][sampleIndexes]
    Y = data[2][1][sampleIndexes]
    # if args.values is not None and args.gen_values != 'data':
    #     X = np.array([args.values])
    # elif args.gen_values == 'zeroes':
    #     X = np.zeros((1, N), dtype=np.float64)
    # elif args.gen_values == 'ones':
    #     X = np.ones((1, N), dtype=np.float64)
    # elif args.gen_values == 'none':
    #     X = np.ones((1, N), dtype=np.float64) * -1
    # elif args.gen_values == 'all':
    #     X = np.array([bitfield(n, N) for n in range(2**N)])
    # else:
    #     raise Exception(f"Unknown generator '{args.gen_values}'")
    print("Input ", X)

    inputCount = X.shape[0]
    # first element in the tuple is delta method, second is monte carlo
    # within the nested tuple, is mean, paramVar, and inputVar
    results: List[List[Tuple[Tensor, Tensor, Tensor]]] = []

    # Delta method
    start_t = perf_counter()
    for i in range(inputCount):
        sample = X[i, :].reshape(1, -1)
        lgc.zero_grad(True)
        cache = EVCache()
        mean, paramVar = deltaMeanAndParameterVariance(psdd, lgc, cache, sample)
        inputVar: Tensor = deltaInputVariance(psdd, lgc, cache, sample, mean)
        results.append([(mean, paramVar, inputVar)])

    end_t = perf_counter()
    print("Delta method took {}".format(end_t - start_t))

    # Monte carlo
    lgc.zero_grad(False)
    start_t = perf_counter()
    params = sampleMonteCarloParameters(lgc, args.samples, randState)
    for i in range(inputCount):
        sample = X[i, :].reshape(1, -1)
        cache = EVCache()
        mean, paramVar, inputVar = monteCarloPrediction(psdd, lgc, params, sample, prefix=f"X {i}")
        results[i].append((mean, paramVar, inputVar))
    print(f"Finished all monte carlo predictions")

    end_t = perf_counter()
    print("Monte carlo took {}".format(end_t - start_t))

    # results
    print("{:<5} {:<20} {:<25} {:<25} {:<20} {:<25} {:<25}".format("Act.", "D Mean", "D pvar", "D ivar", "MC Mean", "MC pvar", "MC ivar"))
    print("")
    for i, result in enumerate(results):
        delta = result[0]
        monteCarlo = result[1]
        print("{:<5} {:<20} {:<25} {:<25} {:<20} {:<25} {:<25}"
              .format(Y[i], delta[0].item(), delta[1].item(), delta[2].item(),
                      monteCarlo[0].item(), monteCarlo[1].item(), monteCarlo[2].item()))
