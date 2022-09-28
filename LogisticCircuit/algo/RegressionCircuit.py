import copy
import gc
import math
from collections import deque
import logging
from time import perf_counter
from typing import Optional, TextIO, List, NoReturn, Tuple, Union, Dict, Set

import numpy as np
import torch
from numpy.random import RandomState

# from .Ridge import Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from .BaseCircuit import BaseCircuit
from .BayesianRegression import BayesianRidge, ARDRegression
from ..structure.AndGate import AndGate, AndChildNode
from ..structure.CircuitNode import OrGate, CircuitTerminal
from ..structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE
from ..structure.Vtree import Vtree
from ..util.DataSet import DataSet

FORMAT = """c variables (from inputs) start from 1
c ids of logistic circuit nodes start from 0
c nodes appear bottom-up, children before parents
c the last line of the file records the bias parameter
c three types of nodes:
c	T (terminal nodes that correspond to true literals)
c	F (terminal nodes that correspond to false literals)
c	D (OR gates)
c
c file syntax:
c Regression Circuit
c T id-of-true-literal-node id-of-vtree variable parameters
c F id-of-false-literal-node id-of-vtree variable parameters
c D id-of-or-gate id-of-vtree number-of-elements (id-of-prime id-of-sub parameters)s
c B bias-parameters
c V covariances
c
"""


# TODO: I feel this should extend LogisticCircuit instead of copying it, or maybe a shared base class
class RegressionCircuit(BaseCircuit):
    def __init__(self, vtree: Vtree, circuit_file: Optional[TextIO] = None,
                 rand_gen: Optional[RandomState] = None, requires_grad: bool = False):
        super().__init__(vtree, 1, circuit_file, rand_gen, requires_grad)

    def _parameter_size(self) -> Union[int, Tuple[int]]:
        return 1

    # Two additional parameters over LogisticCircuit
    def _select_element_and_variable_to_split(self, data: DataSet, num_splits: int,
                                              alpha: float, min_candidate_list: int = 5000) -> List[Tuple[AndGate, int]]:
        # y = self.predict_prob(data.features)
        y: torch.Tensor = self.predict(data.features)

        delta: np.ndarray = (data.labels.reshape(-1, 1) - y).numpy()
        # element_gradients = - 2 * np.dot(data.features.T,  delta) + (2 * alpha * self.parameters).T
        # TODO: we are using torch elsewhere for gradients, feel like here might as well use it later
        element_gradients = (-2 * (delta.reshape(-1, 1) * data.features + (2 * alpha * self.parameters.numpy()))[:, 2 * self._num_variables + 1:])
        # logging.info(
        #     f'e g shape {len(element_gradients)} delta {delta.shape} {data.features.shape}')
        element_gradient_variance = np.var(element_gradients, axis=0)

        # logging.info(
        #     f'\tgradient variances: shape:{element_gradient_variance.shape} max: {np.max(element_gradient_variance)} min: {np.min(element_gradient_variance)}')
        # delta = data.one_hot_labels - y
        # element_gradients = np.stack(
        #     [(delta[:, i].reshape(-1, 1) * data.features)[:, 2 * self._num_variables + 1:]
        #      for i in range(self._num_classes)],
        #     axis=0)
        # element_gradient_variance = np.var(element_gradients, axis=1)
        # element_gradient_variance = np.average(element_gradient_variance, axis=0)

        # print(element_gradient_variance, 'EGV', element_gradient_variance.shape)

        candidates: List[Tuple[AndGate, np.ndarray, np.ndarray]] = sorted(
            zip(self._elements, element_gradient_variance,
                data.features.T[2 * self._num_variables + 1:]),
            reverse=True,
            key=lambda x: x[1],
        )
        selected: List[Tuple[Tuple[AndGate, int], float]] = []
        logging.info(f"{len(list(candidates))} candidates found")
        # print(len(candidates), candidates)
        for (element_to_split, original_variance, original_feature) in candidates[: min(min_candidate_list, len(candidates))]:
            if len(element_to_split.splittable_variables) > 0 and np.sum(original_feature) > 25:
                # logging.info(
                #     f'vars to split {len(element_to_split.splittable_variables)} {np.sum(candidate[2]) > 25}')
                variable_to_split: Optional[int] = None
                min_after_split_variance = float("inf")
                # print('SPLIT', element_to_split.splittable_variables)
                for variable in element_to_split.splittable_variables:
                    variable: int
                    left_feature = original_feature * data.images[:, variable - 1]
                    right_feature = original_feature - left_feature

                    # print('FEATs', left_feature.shape, right_feature.shape)
                    # print('SUM left', np.sum(left_feature), np.sum(right_feature))
                    #
                    #
                    # FIXME: this is hardcoded, disabling it for a moment
                    # if np.sum(left_feature) > 10 and np.sum(right_feature) > 10:

                    # left_gradient = (data.one_hot_labels - y) * left_feature.reshape((-1, 1))
                    # right_gradient = (data.one_hot_labels - y) * right_feature.reshape((-1, 1))
                    left_gradient = -2 * np.dot(left_feature.reshape((-1, 1)).T, delta) + (2 * alpha * self.parameters.numpy()).T
                    right_gradient = -2 * \
                        np.dot(right_feature.reshape((-1, 1)).T, delta) + \
                        (2 * alpha * self.parameters.numpy()).T

                    # logging.info(f'LG {left_gradient.shape} RG {right_gradient.shape}')
                    w = np.sum(data.images[:, variable - 1]) / data.num_samples

                    after_split_variance = w * np.average(np.var(left_gradient, axis=0)) + (1 - w) * np.average(
                        np.var(right_gradient, axis=0)
                    )
                    if after_split_variance < min_after_split_variance:
                        min_after_split_variance = after_split_variance
                        variable_to_split = variable
                # print('VARS', min_after_split_variance, 'ORIG', original_variance)
                if min_after_split_variance < original_variance:
                    assert variable_to_split is not None
                    improved_amount = min_after_split_variance - original_variance
                    # print('imp amount', improved_amount, len(selected))
                    if len(selected) == num_splits:
                        if improved_amount < selected[0][1]:
                            selected = selected[1:]
                            selected.append(((element_to_split, variable_to_split), improved_amount))
                            selected.sort(key=lambda x: x[1])
                    else:
                        selected.append(((element_to_split, variable_to_split), improved_amount))
                        selected.sort(key=lambda x: x[1])

        gc.collect()
        splits = [x[0] for x in selected]
        logging.info(f"{len(splits)} splits found")
        return splits

    # Same as LogisticCircuit except the parameter copying
    def _record_learned_parameters(self, parameters: np.ndarray, requires_grad: bool = False) -> NoReturn:
        # TODO: reshape required?
        self._parameters = torch.tensor(parameters, requires_grad=requires_grad).reshape(1, -1)
        # print("todo fix the _record_learned_parameters")
        # print('PARAMS', self._parameters.shape)
        self.set_node_parameters(self._parameters)

    # def calculate_accuracy(self, data):
    #     """Calculate accuracy given the learned parameters on the provided data."""
    #     y = self.predict(data.features)
    #     accuracy = np.sum(y == data.labels) / data.num_samples
    #     return accuracy


    # def predict(self, features):
    #     y = self.predict_prob(features)
    #     return np.argmax(y, axis=1)

    def predict(self, features: np.ndarray) -> torch.Tensor:
        return torch.mm(torch.from_numpy(features), self._parameters.T)

    # def predict_prob(self, features):
    #     """Predict the given images by providing their corresponding features."""
    #     y = 1.0 / (1.0 + np.exp(-np.dot(features, self._parameters.T)))
    #     return y

    def learn_parameters(self, data: DataSet, num_iterations: int, num_cores: int = -1,
                         solver: str = "auto", alpha: float = 1.0,
                         rand_gen: Union[int, RandomState, None] = None,
                         params: Dict = None, cv_params: Dict = None) -> NoReturn:
        """Logistic Psdd's parameter learning is reduced to logistic regression.
        We use mini-batch SGD to optimize the parameters."""
        if params is None:
            params = {}

        if solver == 'bayesian-ridge':
            model = BayesianRidge(
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                n_iter=num_iterations,
                # coef_init=self._parameters.flatten().numpy(),
                # random_state=rand_gen, TODO?
                **params
            )
        elif solver == 'bayesian-ard':
            model = ARDRegression(
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                n_iter=num_iterations,
                # coef_init=self._parameters.flatten().numpy(),
                # random_state=rand_gen, TODO?
                **params
            )
        # default to ridge and pass along solver
        else:
            model = Ridge(
                alpha=alpha,
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                max_iter=num_iterations,
                solver=solver,
                # coef_=self._parameters,
                random_state=rand_gen,
                **params
            )

        # grid search if given grid search params
        if cv_params is not None:
            model = GridSearchCV(model, cv_params, cv=min(5, data.labels.shape[0]), n_jobs=num_cores)

        print("About to fit model")
        model.fit(data.features, data.labels.numpy())

        # extract best model from grid search
        if cv_params is not None:
            print("Best params: ", model.best_params_)
            model = model.best_estimator_

        # bayesian variants store the covariance
        if solver in ('bayesian-ridge', 'bayesian-ard'):
            w, v = np.linalg.eig(model.sigma_)
            print("Score:", model.score(data.features, data.labels.numpy()))
            print("Covariance:", np.sum(w), np.shape(model.sigma_))
            self._covariance = [model.sigma_]
        # print('PARAMS', self._parameters.shape, model.coef_.shape)

        self._record_learned_parameters(model.coef_)
        gc.collect()

    def change_structure(self, data: DataSet, depth: int, num_splits: int, alpha: float) -> NoReturn:
        splits: List[Tuple[AndGate, int]]  = self._select_element_and_variable_to_split(data, num_splits, alpha)
        # print(len(splits), 'SPLITS')
        for element_to_split, variable_to_split in splits:
            if not element_to_split.flag:
                self._split(element_to_split, variable_to_split, depth)
        self._serialize()

    # Changes one string from LogisticCircuit
    def save(self, f: TextIO) -> NoReturn:
        self._serialize()
        f.write(FORMAT)
        f.write(f"Regression Circuit\n")
        for terminal_node in self._terminal_nodes:
            terminal_node.save(f)
        for decision_node in reversed(self._decision_nodes):
            decision_node.save(f)
        f.write("B")
        for parameter in self._bias:
            f.write(f" {parameter}")
        f.write("\n")
        if self._covariance is not None:
            for covMatrix in self._covariance:
                f.write("V")
                for cov in covMatrix.flatten():
                    f.write(f" {cov}")
                f.write("\n")

    # Bit different, should compare if considering merging
    def load(self, f: TextIO) -> OrGate:
        # read the format at the beginning
        line = f.readline()
        while line[0] == "c":
            line = f.readline()

        # serialize the vtree
        vtree_nodes: Dict[int, Vtree] = dict()
        unvisited_vtree_nodes: deque[Vtree] = deque()
        unvisited_vtree_nodes.append(self._vtree)
        while len(unvisited_vtree_nodes):
            node: Vtree = unvisited_vtree_nodes.popleft()
            vtree_nodes[node.index] = node
            if not node.is_leaf():
                unvisited_vtree_nodes.append(node.left)
                unvisited_vtree_nodes.append(node.right)

        # extract the saved logistic circuit
        terminal_nodes: Optional[Dict[int, Tuple[CircuitTerminal, Set[int]]]]  = dict()
        line = f.readline()
        while line[0] == "T" or line[0] == "F":
            line_as_list = line.strip().split(" ")
            positive_literal, var = (line_as_list[0] == "T"), int(line_as_list[3])
            index, vtree_index = int(line_as_list[1]), int(line_as_list[2])
            parameters: List[float] = []
            # for i in range(self._num_classes):
            # parameters.append(float(line_as_list[4 + i]))
            parameters.append(float(line_as_list[4]))
            if positive_literal:
                terminal_nodes[index] = (
                    CircuitTerminal(index, vtree_nodes[vtree_index], var, LITERAL_IS_TRUE,
                                    torch.tensor(parameters, dtype=torch.float64)),
                    {var}
                )
            else:
                terminal_nodes[index] = (
                    CircuitTerminal(index, vtree_nodes[vtree_index], var, LITERAL_IS_FALSE,
                                    torch.tensor(parameters, dtype=torch.float64)),
                    {-var})
            self._largest_index = max(self._largest_index, index)
            line = f.readline()

        self._terminal_nodes = [x[0] for x in terminal_nodes.values()]
        self._terminal_nodes.sort(key=lambda x: (-x.var_value, x.var_index))

        if len(self._terminal_nodes) != 2 * self._num_variables:
            raise ValueError(
                "Number of terminal nodes recorded in the circuit file "
                "does not match 2 * number of variables in the provided vtree."
            )

        # Broaden type hints from circuit terminal to circuit node
        nodes: Dict[int, Tuple[AndChildNode, Set[int]]] = terminal_nodes
        terminal_nodes = None

        root: Optional[OrGate] = None
        while line[0] == "D":
            line_as_list = line.strip().split(" ")
            index, vtree_index, num_elements = int(line_as_list[1]), int(line_as_list[2]), int(line_as_list[3])
            elements: List[AndGate] = []
            variables: Set[int] = set()
            for i in range(num_elements):
                # prime_index = int(line_as_list[i * (self._num_classes + 2) + 4].strip('('))
                # sub_index = int(line_as_list[i * (self._num_classes + 2) + 5])
                #
                # FIXME: remove constants
                prime_index = int(line_as_list[i * (1 + 2) + 4].strip("("))
                sub_index = int(line_as_list[i * (1 + 2) + 5])
                element_variables = nodes[prime_index][1].union(nodes[sub_index][1])
                variables = variables.union(element_variables)
                splittable_variables: Set[int] = set()
                for variable in element_variables:
                    if -variable in element_variables:
                        splittable_variables.add(abs(variable))
                parameters: List[float] = []
                # for j in range(self._num_classes):
                #     parameters.append(
                #         float(line_as_list[i * (self._num_classes + 2) + 6 + j].strip(')')))
                parameters.append(float(line_as_list[i * (1 + 2) + 6].strip(")")))
                elements.append(AndGate(nodes[prime_index][0], nodes[sub_index][0], torch.tensor(parameters, dtype=torch.float64)))
                elements[-1].splittable_variables = splittable_variables
            root = OrGate(index, vtree_nodes[vtree_index], elements)
            nodes[index] = (root, variables)
            self._largest_index = max(self._largest_index, index)
            line = f.readline()

        # Ensure the file contained at least one decision node
        if root is None:
            raise ValueError("Circuit must have at least one decision node to represent the root node")

        if line[0] != "B":
            raise ValueError("After decision nodes in a circuit must record the bias parameters.")
        self._bias = torch.tensor([float(x) for x in line.strip().split(" ")[1:]], dtype=torch.float64)

        # the parameters vector will be reconstructed after loading, but the covariance will need to be read
        line = f.readline()
        covariances: List[np.ndarray] = []
        while line and line[0] == "V":
            vector = np.array([float(x) for x in line.strip().split(" ")[1:]], dtype=np.float64)
            matrixSize: Union[int, float] = math.sqrt(vector.size)
            if not matrixSize.is_integer():
                print("Error: covariance matrix must be square")
                # raise ValueError("Covariance matrix must be square"
            else:
                matrixSize = int(matrixSize)
                covariances.append(vector.reshape((matrixSize, matrixSize)))
            line = f.readline()
        self._covariance = covariances

        del nodes
        gc.collect()
        return root


def learn_regression_circuit(
    vtree: Vtree,
    train: DataSet,
    valid: Optional[DataSet] = None,
    solver: str = "auto",
    max_iter_sl: int = 1000,
    max_iter_pl: int = 1000,
    depth: int = 20,
    num_splits: int = 10,
    alpha: float = 0.2,
    validate_every: int = 10,
    patience: int = 2,
    rand_gen: Optional[RandomState] = None
) -> Tuple[RegressionCircuit, List[float]]:

    # #
    # # FIXEME: do we need this?
    # X[np.where(X == 0.0)[0]] = 1e-5
    # X[np.where(X == 1.0)[0]] -= 1e-5

    # train = Dataset(train_x, train_y)

    error_history: List[float] = []

    circuit = RegressionCircuit(vtree, rand_gen=rand_gen)
    train.features = circuit.calculate_features(train.images)

    logging.info(f"The starting circuit has {circuit.num_parameters} parameters.")
    train.features = circuit.calculate_features(train.images)
    train_acc: float = circuit.calculate_error(train)
    logging.info(f" error: {train_acc:.5f}")
    error_history.append(train_acc)

    logging.info("Start structure learning.")
    sl_start_t: float = perf_counter()

    valid_best = +np.inf
    best_model = copy.deepcopy(circuit)
    c_patience = 0
    for i in range(max_iter_sl):
        cur_time = perf_counter()

        circuit.change_structure(train, depth, num_splits, alpha)

        train.features = circuit.calculate_features(train.images)
        pl_start_t = perf_counter()
        circuit.learn_parameters(train, max_iter_pl, rand_gen=rand_gen, solver=solver)
        pl_end_t = perf_counter()

        train_acc: float = circuit.calculate_error(train)

        logging.info(f"done iter {i+1}/{max_iter_sl} in {perf_counter() - cur_time} secs")
        logging.info(f"\tcircuit size: {circuit.num_parameters}")
        logging.info(f"\tparameters learning done in {pl_end_t - pl_start_t} secs")
        logging.info(f"\terror: {train_acc:.5f} size {circuit.num_parameters}")
        error_history.append(train_acc)

        if i % validate_every == 0 and valid is not None:
            logging.info(f"evaluate in the validation set")
            valid.features = circuit.calculate_features(valid.images)
            valid_err = circuit.calculate_error(valid)
            if valid_err >= valid_best:
                logging.info(f"Worsening on valid: {valid_err} > prev best {valid_best}")
                if c_patience >= patience:
                    logging.info(f"Exceeding patience {c_patience} >= {patience}: STOP")
                    break
                else:
                    c_patience += 1
            else:
                logging.info(f'Found new best model {valid_err}')
                best_model = copy.deepcopy(circuit)
                valid_best = valid_err
                c_patience = 0

    sl_end_t: float = perf_counter()
    logging.info(f"Structure learning done in {sl_end_t - sl_start_t} secs")

    return best_model, error_history
