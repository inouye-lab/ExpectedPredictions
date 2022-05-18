import sys

import argparse
from typing import List, Tuple

import numpy as np
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
    parser.add_argument('--values', type=int, nargs='+', help='Input to the model')
    parser.add_argument('--gen_values', type=str, default='zeroes', help='Generates a vector of values')

    parser.add_argument("--classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--variables", type=int, required=True,
                        help="Number of variables in the dataset")
    parser.add_argument("--samples", type=int, required=True,
                        help="Number of monte carlo samples")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Number of monte carlo samples")
    #
    # parsing the args
    args = parser.parse_args()

    FOLDER = args.model
    VTREE_FILE = FOLDER + ".vtree"
    GLC_FILE = FOLDER + ".glc"
    PSDD_FILE = FOLDER + ".psdd"
    CLASSES = args.classes
    N = args.variables
    
    lc_vtree = LC_Vtree.read(VTREE_FILE)
    with open(GLC_FILE) as circuit_file:
        lgc = LogisticCircuit(lc_vtree, CLASSES, circuit_file=circuit_file, requires_grad=True)
        
    print("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    if args.values is not None:
        X = np.array([args.values])
    elif args.gen_values == 'zeroes':
        X = np.zeros((1, N), dtype=np.float64)
    elif args.gen_values == 'ones':
        X = np.ones((1, N), dtype=np.float64)
    elif args.gen_values == 'none':
        X = np.ones((1, N), dtype=np.float64) * -1
    elif args.gen_values == 'all':
        X = np.array([bitfield(n, N) for n in range(2**N)])
    else:
        raise Exception(f"Unknown generator '{args.gen_values}'")
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
    params = sampleMonteCarloParameters(lgc, args.samples, RandomState(args.seed))
    for i in range(inputCount):
        sample = X[i, :].reshape(1, -1)
        cache = EVCache()
        mean, paramVar, inputVar = monteCarloPrediction(psdd, lgc, params, sample, prefix=f"X {i}")
        results[i].append((mean, paramVar, inputVar))
    print(f"Finished all monte carlo predictions")

    end_t = perf_counter()
    print("Monte carlo took {}".format(end_t - start_t))

    # results
    print("{:<20} {:<25} {:<25} {:<20} {:<25} {:<25}".format("D Mean", "D pvar", "D ivar", "MC Mean", "MC pvar", "MC ivar"))
    print("")
    for result in results:
        delta = result[0]
        monteCarlo = result[1]
        print("{:<20} {:<25} {:<25} {:<20} {:<25} {:<25}"
              .format(delta[0].item(), delta[1].item(), delta[2].item(),
                      monteCarlo[0].item(), monteCarlo[1].item(), monteCarlo[2].item()))
