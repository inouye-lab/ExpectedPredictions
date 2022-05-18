import sys

import argparse
import numpy as np
import torch

from LogisticCircuit.structure.Vtree import Vtree as LC_Vtree

from collections import defaultdict

import pdb

from pypsdd.vtree import Vtree as PSDD_Vtree
from pypsdd.manager import PSddManager
import pypsdd.psdd_io
from pypsdd.data import Inst, InstMap


import itertools
from LogisticCircuit.algo.LogisticCircuit import LogisticCircuit

import circuit_expect
from sympy import *
from utils import *

from scipy.special import logit
from scipy.special import expit

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def taylor_aprox(psdd, lgc, n):
    sum = 0.5
    coeff = [1 / 4.0, -1 / 48.0, 1 / 480.0, -17 / 80640.0, 31 / 1451520.0, -691 / 319334400.0,
             5461 / 24908083200.0, -929569 / 41845579776000.0]#, 3202291 /
             #1422749712384000.0, -221930581 / 973160803270656000.0,
             #4722116521 / 204363768686837760000.0]
    for k, c in enumerate(coeff):
        cur = brute_force_expectation(psdd, lgc, n, k=2 * k + 1, compute_prob=False)
        sum += c * cur
    return sum


"""Creates a bit field as an array for the given number"""
def bitfield(number: int, size: int):
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
    #
    # parsing the args
    args = parser.parse_args()

    FOLDER = args.model
    VTREE_FILE = FOLDER + ".vtree"
    GLC_FILE = FOLDER + ".glc"
    PSDD_FILE = FOLDER + ".psdd"
    CLASSES = args.classes
    N = args.variables

    # FOLDER = "notebooks/rand-gen-grid/exp-D9-N500-C6-B1/D9-N500-C6-B1"
    # VTREE_FILE = FOLDER + ".vtree"
    # GLC_FILE = FOLDER + ".glc"
    # PSDD_FILE = FOLDER + ".psdd"
    # CLASSES = 6
    # N = 9

    # VTREE_FILE = "test/circuits/5.vtree"
    # GLC_FILE = "test/circuits/5.glc"
    # PSDD_FILE = "test/circuits/5.psdd"
    # CLASSES = 2
    # N = 2

    # VTREE_FILE = "notebooks/exp-D15-N1000-C4-balanced.vtree"
    # GLC_FILE = "notebooks/exp-D15-N1000-C4.glc"
    # PSDD_FILE = "notebooks/exp-D15-N1000-C4.psdd"
    # CLASSES = 4
    # N = 15

    # VTREE_FILE = "exp/test-adult/adult-test-I/adult.vtree"
    # GLC_FILE   = "exp/test-adult/adult-test-I/adult.glc"
    # PSDD_FILE  = "exp/test-adult/adult-test-I/adult.psdd"
    # CLASSES = 2
    # N = 157

    
    # VTREE_FILE = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/elevators.vtree"
    # GLC_FILE   = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/best/elevators.glc"
    # PSDD_FILE  = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/best/elevators.psdd"
    # CLASSES = 1
    # N = 182

    # VTREE_FILE = "exp/mnist-final/mnist.vtree"
    # GLC_FILE = "exp/mnist-final/mnist.glc"
    # PSDD_FILE = "exp/mnist-final/mnist.psdd"
    # CLASSES = 10
    # N = 28*28
    
    lc_vtree = LC_Vtree.read(VTREE_FILE)
    with open(GLC_FILE) as circuit_file:
        lgc = LogisticCircuit(lc_vtree, CLASSES, circuit_file=circuit_file, requires_grad=True)
        
    print("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = pypsdd.psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    try:
        from time import perf_counter
    except:
        from time import time
        perf_counter = time

    if args.values is not None:
        X = np.array([args.values])
    elif args.gen_values == 'zeroes':
        X = np.zeros((1, N), dtype=np.float64)
    elif args.gen_values == 'ones':
        X = np.ones((1, N), dtype=np.float64)
    elif args.gen_values == 'all':
        X = np.array([bitfield(n, N) for n in range(2**N)])
    else:
        raise Exception(f"Unknown generator '{args.gen_values}'")
    print("Input ", X)

    start_t = perf_counter()
    cache = EVCache()
    lgc.zero_grad()
    ans = circuit_expect.Expectation(psdd, lgc, cache, X)
    print(ans)
    ans.backward(torch.ones((X.shape[0], CLASSES), dtype=torch.float))
    print(lgc.parameters.grad)

    # lgc.zero_grad()
    # cache = EVCache()
    # ans = circuit_expect.Expectation(psdd, lgc, cache, X)
    # print(ans)
    # ans.backward(torch.ones((X.shape[0], CLASSES), dtype=torch.float))
    # print(lgc.parameters.grad)

    end_t = perf_counter()

    print("Time taken {}".format(end_t - start_t))