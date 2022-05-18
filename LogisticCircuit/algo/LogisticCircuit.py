import copy
import gc
from collections import deque
import logging
from time import perf_counter

import numpy as np
import torch
from numpy.random import RandomState

from ..algo.LogisticRegression import LogisticRegression
from ..algo.BayesianRegression import BayesianRidge
from ..algo.BayesianLogistic import EBLogisticRegression, VBLogisticRegression
from ..structure.AndGate import AndGate, AndChildNode
from ..structure.CircuitNode import OrGate, CircuitTerminal
from ..structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE
from ..structure.Vtree import Vtree
from ..util.DataSet import DataSet

from typing import List, Dict, Set, Tuple, Optional, Union, TextIO, NoReturn


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
c Logisitic Circuit
c T id-of-true-literal-node id-of-vtree variable parameters
c F id-of-false-literal-node id-of-vtree variable parameters
c D id-of-or-gate id-of-vtree number-of-elements (id-of-prime id-of-sub parameters)s
c B bias-parameters
c
"""


class LogisticCircuit(object):
    _vtree: Vtree
    _num_classes: int
    _largest_index: int
    _num_variables: int

    rand_gen: RandomState

    _terminal_nodes: List[Optional[CircuitTerminal]]
    _decision_nodes: Optional[List[OrGate]]
    _elements: Optional[List[AndGate]]
    _parameters: Optional[torch.Tensor]
    _bias: torch.Tensor
    _root: OrGate

    def __init__(self, vtree: Vtree, num_classes: int, circuit_file: Optional[TextIO] = None,
                 rand_gen: Optional[RandomState] = None):
        self._vtree = vtree
        self._num_classes = num_classes
        self._largest_index = 0
        self._num_variables = vtree.var_count

        if rand_gen is None:
            rand_gen = np.random.RandomState(1337)
        self.rand_gen = rand_gen

        self._terminal_nodes = [None] * 2 * self._num_variables
        self._decision_nodes = None
        self._elements = None
        self._parameters = None
        self._bias = torch.tensor(self.rand_gen.random_sample(size=(num_classes,)))  # TODO: copy needed?

        if circuit_file is None:
            self._generate_all_terminal_nodes(vtree)
            self._root = self._new_logistic_psdd(vtree)
        else:
            self._root = self.load(circuit_file)

        self._serialize()

    @property
    def vtree(self) -> Vtree:
        return self._vtree

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_variables(self) -> int:
        return self._num_variables

    @property
    def root(self) -> OrGate:
        return self._root

    @property
    def num_parameters(self) -> int:
        # numpy return self._parameters.size
        return self._parameters.size()

    @property
    def parameters(self) -> Optional[torch.Tensor]:
        return self._parameters

    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    def _generate_all_terminal_nodes(self, vtree: Vtree) -> NoReturn:
        if vtree.is_leaf():
            var_index = vtree.var
            self._terminal_nodes[var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_TRUE,
                torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,)))  # TODO: can we directly generate using torch?
            )
            self._largest_index += 1
            self._terminal_nodes[self._num_variables + var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_FALSE,
                torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,)))  # TODO: can we directly generate using torch?
            )
            self._largest_index += 1
        else:
            self._generate_all_terminal_nodes(vtree.left)
            self._generate_all_terminal_nodes(vtree.right)

    def _new_logistic_psdd(self, vtree: Vtree) -> OrGate:
        left_vtree: Vtree = vtree.left
        right_vtree: Vtree = vtree.right
        prime_variable: int = left_vtree.var
        sub_variable: int = right_vtree.var
        elements: List[AndGate] = list()
        if left_vtree.is_leaf() and right_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._terminal_nodes[sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
        elif left_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._new_logistic_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._new_logistic_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(right_vtree.variables)
        elif right_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._terminal_nodes[sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(left_vtree.variables)
        else:
            elements.append(
                AndGate(
                    self._new_logistic_psdd(left_vtree),
                    self._new_logistic_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=(self._num_classes,))),
                )
            )
            elements[0].splittable_variables = copy.deepcopy(vtree.variables)
        root = OrGate(self._largest_index, vtree, elements)
        self._largest_index += 1
        return root

    def _serialize(self) -> NoReturn:
        """Serialize all the decision nodes in the logistic psdd.
           Serialize all the elements in the logistic psdd. """
        self._decision_nodes = [self._root]
        self._elements = []
        decision_node_indices = set()
        decision_node_indices.add(self._root.index)
        unvisited: deque[OrGate] = deque()
        unvisited.append(self._root)
        while len(unvisited) > 0:
            current = unvisited.popleft()
            for element in current.elements:
                self._elements.append(element)
                element.flag = False
                if isinstance(element.prime, OrGate) and element.prime.index not in decision_node_indices:
                    decision_node_indices.add(element.prime.index)
                    self._decision_nodes.append(element.prime)
                    unvisited.append(element.prime)
                if isinstance(element.sub, OrGate) and element.sub.index not in decision_node_indices:
                    decision_node_indices.add(element.sub.index)
                    self._decision_nodes.append(element.sub)
                    unvisited.append(element.sub)
        self._parameters = self._bias.reshape(-1, 1)
        for terminal_node in self._terminal_nodes:
            # print(self._parameters.shape, terminal_node.parameter.reshape(-1, 1).shape)
            self._parameters = torch.cat((self._parameters, terminal_node.parameter.reshape(-1, 1)), dim=1)
        for element in self._elements:
            self._parameters = torch.cat((self._parameters, element.parameter.reshape(-1, 1)), dim=1)
        gc.collect()

    def _record_learned_parameters(self, parameters: np.array, requires_grad: bool = False) -> NoReturn:
        # hack for the binary classification case in sklearn
        # FIXME: generalize this
        # TODO: this the right way to copy the parameters into a tensor?
        if self._num_classes == 2 and not requires_grad:
            self._parameters = torch.vstack((
                torch.zeros(parameters.shape, requires_grad=requires_grad),
                torch.tensor(parameters, requires_grad=requires_grad)
            ))
        else:
            self._parameters = torch.tensor(parameters, requires_grad=requires_grad)  # copy.deepcopy(parameters)
        self._bias = self._parameters[:, 0]
        for i in range(len(self._terminal_nodes)):
            self._terminal_nodes[i].parameter = self._parameters[:, i + 1]
        for i in range(len(self._elements)):
            self._elements[i].parameter = self._parameters[:, i + 1 + 2 * self._num_variables]
        gc.collect()

    def calculate_features(self, images: np.ndarray) -> np.ndarray:
        num_images: int = images.shape[0]
        for terminal_node in self._terminal_nodes:
            terminal_node.calculate_prob(images)
        for decision_node in reversed(self._decision_nodes):
            decision_node.calculate_prob()
        self._root.feature = np.ones(shape=(num_images,), dtype=np.float64)
        for decision_node in self._decision_nodes:
            decision_node.calculate_feature()
        # bias feature
        bias_features = np.ones(shape=(num_images,), dtype=np.float64)
        terminal_node_features = np.vstack(
            [terminal_node.feature for terminal_node in self._terminal_nodes])
        element_features = np.vstack([element.feature for element in self._elements])
        features = np.vstack((bias_features, terminal_node_features, element_features))
        for terminal_node in self._terminal_nodes:
            terminal_node.feature = None
            terminal_node.prob = None
        for element in self._elements:
            element.feature = None
            element.prob = None
        return features.T

    def _select_element_and_variable_to_split(self, data: DataSet, num_splits: int) -> List[Tuple[AndGate, int]]:
        y: torch.Tensor = self.predict_prob(data.features)

        delta = data.one_hot_labels - y
        element_gradients = np.stack(
            [
                (delta[:, i].reshape(-1, 1) * data.features)[:, 2 * self._num_variables + 1:]
                for i in range(self._num_classes)
            ],
            axis=0,
        )
        element_gradient_variance = np.var(element_gradients, axis=1)
        element_gradient_variance = np.average(element_gradient_variance, axis=0)

        # print(element_gradient_variance, 'EGV')

        candidates: List[Tuple[AndGate, np.ndarray, np.ndarray]] = sorted(
            zip(self._elements, element_gradient_variance,
                data.features.T[2 * self._num_variables + 1:]),
            reverse=True,
            key=lambda x: x[1],
        )
        selected: List[Tuple[Tuple[AndGate, int], float]] = []
        # print(len(candidates), candidates)
        for (element_to_split, original_variance, original_feature) in candidates[: min(5000, len(candidates))]:
            if len(element_to_split.splittable_variables) > 0 and np.sum(original_feature) > 25:
                variable_to_split: Optional[int] = None
                min_after_split_variance = float("inf")
                # print('SPLIT', element_to_split.splittable_variables)
                for variable in element_to_split.splittable_variables:
                    variable: int
                    left_feature = original_feature * data.images[:, variable - 1]
                    right_feature = original_feature - left_feature

                    # print('SUM left', np.sum(left_feature), np.sum(right_feature))
                    #
                    #
                    # FIXME: this is hardcoded, disabling it for a moment
                    # if np.sum(left_feature) > 10 and np.sum(right_feature) > 10:

                    left_gradient = (data.one_hot_labels - y) * left_feature.reshape((-1, 1))
                    right_gradient = (data.one_hot_labels - y) * right_feature.reshape((-1, 1))

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
        return [x[0] for x in selected]

    def _split(self, element_to_split: AndGate, variable_to_split: int, depth: int) -> NoReturn:
        parent = element_to_split.parent
        original_element, copied_element = self._copy_and_modify_element_for_split(
            element_to_split, variable_to_split, 0, depth
        )
        if original_element is None or copied_element is None:
            raise ValueError("Split elements become invalid.")
        parent.add_element(copied_element)

    def _copy_and_modify_element_for_split(self, original_element: AndGate, variable: int, current_depth: int,
                                           max_depth: int) -> Tuple[Optional[AndGate], Optional[AndGate]]:
        original_element.flag = True
        original_element.remove_splittable_variable(variable)
        original_prime: AndChildNode = original_element.prime
        original_sub: AndChildNode = original_element.sub
        copied_element: Optional[AndGate]
        if current_depth >= max_depth:
            if variable in original_prime.vtree.variables:
                original_prime, copied_prime = self._copy_and_modify_node_for_split(
                    original_prime, variable, current_depth, max_depth
                )
                copied_sub = original_sub
            elif variable in original_sub.vtree.variables:
                original_sub, copied_sub = self._copy_and_modify_node_for_split(
                    original_sub, variable, current_depth, max_depth
                )
                copied_prime = original_prime
            else:
                copied_prime = original_prime
                copied_sub = original_sub
        else:
            original_prime, copied_prime = self._copy_and_modify_node_for_split(
                original_prime, variable, current_depth, max_depth
            )
            original_sub, copied_sub = self._copy_and_modify_node_for_split(
                original_sub, variable, current_depth, max_depth)
        if copied_prime is not None and copied_sub is not None:
            copied_element = AndGate(copied_prime, copied_sub,
                                     original_element.parameter.clone()) # copy.deepcopy(original_element.parameter))
            copied_element.splittable_variables = copy.deepcopy(
                original_element.splittable_variables)
        else:
            copied_element = None
        if original_prime is not None and original_sub is not None:
            original_element.prime = original_prime
            original_element.sub = original_sub
        else:
            original_element = None
        return original_element, copied_element

    def _copy_and_modify_node_for_split(self, original_node: AndChildNode, variable: int, current_depth: int,
                                        max_depth: int) -> Tuple[Optional[AndChildNode], Optional[AndChildNode]]:
        if original_node.num_parents == 0:
            raise ValueError("Some node does not have a parent.")
        original_node.decrease_num_parents_by_one()
        copied_node: Optional[AndChildNode]
        if isinstance(original_node, CircuitTerminal):
            if original_node.var_index == variable:
                if original_node.var_value == LITERAL_IS_TRUE:
                    copied_node = None
                elif original_node.var_value == LITERAL_IS_FALSE:
                    original_node = None
                    copied_node = self._terminal_nodes[self._num_variables + variable - 1]
                else:
                    raise ValueError(
                        "Under the current setting,"
                        "we only support terminal nodes that are either positive or negative literals."
                    )
            else:
                copied_node = original_node
            return original_node, copied_node
        else:
            if original_node.num_parents > 0:
                original_node = self._deep_copy_node(
                    original_node, variable, current_depth, max_depth)
            copied_elements = []
            i = 0
            while i < len(original_node.elements):
                original_element, copied_element = self._copy_and_modify_element_for_split(
                    original_node.elements[i], variable, current_depth + 1, max_depth
                )
                if original_element is None:
                    original_node.remove_element(i)
                else:
                    i += 1
                if copied_element is not None:
                    copied_elements.append(copied_element)
            if len(copied_elements) == 0:
                copied_node = None
            else:
                self._largest_index += 1
                copied_node = OrGate(self._largest_index, original_node.vtree, copied_elements)
            if len(original_node.elements) == 0:
                original_node = None
            return original_node, copied_node

    def _deep_copy_node(self, node: AndChildNode, variable: int, current_depth: int, max_depth: int) -> AndChildNode:
        if isinstance(node, CircuitTerminal):
            return node
        else:
            if len(node.elements) == 0:
                raise ValueError("Decision nodes should have at least one elements.")
            copied_elements = []
            for element in node.elements:
                copied_elements.append(self._deep_copy_element(
                    element, variable, current_depth + 1, max_depth))
            self._largest_index += 1
            return OrGate(self._largest_index, node.vtree, copied_elements)

    def _deep_copy_element(self, element: AndGate, variable: int, current_depth: int, max_depth: int) -> AndGate:
        copied_element: AndGate
        if current_depth >= max_depth:
            if variable in element.prime.vtree.variables:
                copied_element = AndGate(
                    self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                    element.sub,
                    element.parameter.clone(),  # copy.deepcopy(element.parameter),
                )
            elif variable in element.sub.vtree.variables:
                copied_element = AndGate(
                    element.prime,
                    self._deep_copy_node(element.sub, variable, current_depth, max_depth),
                    element.parameter.clone(),  # copy.deepcopy(element.parameter),
                )
            else:
                copied_element = AndGate(element.prime, element.sub,
                                         element.parameter.clone(),  # copy.deepcopy(element.parameter)
                                         )
        else:
            copied_element = AndGate(
                self._deep_copy_node(element.prime, variable, current_depth, max_depth),
                self._deep_copy_node(element.sub, variable, current_depth, max_depth),
                element.parameter.clone(),  # copy.deepcopy(element.parameter),
            )
        copied_element.splittable_variables = copy.deepcopy(element.splittable_variables)
        return copied_element

    def calculate_accuracy(self, data: DataSet) -> float:
        """Calculate accuracy given the learned parameters on the provided data."""
        y = self.predict(data.features)
        accuracy = torch.div(torch.sum(y.eq(data.labels)), data.num_samples)
        return accuracy.item()

    def predict(self, features: np.ndarray) -> torch.Tensor:
        y = self.predict_prob(features)
        return torch.argmax(y, dim=1)

    def predict_prob(self, features: np.ndarray) -> torch.Tensor:
        """Predict the given images by providing their corresponding features."""
        # For some reason tensor + 1 is a float in pycharm, but torch.add resolves the correct type hint
        y: torch.Tensor = 1.0 / torch.exp(-torch.mm(torch.from_numpy(features), self._parameters.T)).add(1)
        print(y.shape[0])
        if y.shape[1] == 1:
            ans = torch.zeros((y.shape[0], 2), dtype=torch.float)
            ans[:, 1] = y[:, 0]
            ans[:, 0] = 1.0 - y[:, 0]

            print(ans)
            return ans
        return y

    def learn_parameters(self, data: DataSet, num_iterations: int, num_cores: int = -1, solver: str = "auto",
                         C: Union[List[float], int] = 10, rand_gen: Union[int, RandomState, None] = None) -> NoReturn:
        """Logistic Psdd's parameter learning is reduced to logistic regression.
        We use mini-batch SGD to optimize the parameters."""
        if solver == 'variational-bayes':
            model = VBLogisticRegression(
                fit_intercept=False,
                n_iter=num_iterations,
                #C=C,
                tol=1e-5,
                n_jobs=num_cores,
                coef=self.parameters.numpy()
            )
        elif solver == 'empirical-bayes':
            model = EBLogisticRegression(
                fit_intercept=False,
                n_iter=num_iterations,
                #C=C,
                tol=1e-5,
                n_jobs=num_cores,
                coef=self.parameters.numpy()
            )
        else:
            # original work used saga, treat as default
            if solver == "auto":
                solver = "saga"
            model = LogisticRegression(
                solver=solver,
                fit_intercept=False,
                multi_class="ovr",
                max_iter=num_iterations,
                C=C,
                warm_start=True,
                tol=1e-5,
                coef_=self._parameters.numpy(),
                n_jobs=num_cores,
                random_state=rand_gen
            )
        print("About to fit model")
        model.fit(data.features, data.labels.numpy())
        # todo: empirical-bayes does not get a matrix of variances, just a vector
        if solver in ('variational-bayes', 'empirical-bayes'):
            print("Covariance:", np.sum(model.sigma_), np.shape(model.sigma_))
        self._record_learned_parameters(model.coef_)
        gc.collect()

    def change_structure(self, data: DataSet, depth: int, num_splits: int) -> NoReturn:
        splits: List[Tuple[AndGate, int]] = self._select_element_and_variable_to_split(data, num_splits)
        print(len(splits), "SPLITS")
        for element_to_split, variable_to_split in splits:
            if not element_to_split.flag:
                self._split(element_to_split, variable_to_split, depth)
        self._serialize()

    def save(self, f: TextIO) -> NoReturn:
        self._serialize()
        f.write(FORMAT)
        f.write(f"Logisitic Circuit\n")
        for terminal_node in self._terminal_nodes:
            terminal_node.save(f)
        for decision_node in reversed(self._decision_nodes):
            decision_node.save(f)
        f.write("B")
        for parameter in self._bias:
            f.write(f" {parameter}")
        f.write("\n")

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

        # extract the terminal nodes
        terminal_nodes: Optional[Dict[int, Tuple[CircuitTerminal, Set[int]]]] = dict()
        line = f.readline()
        while line[0] == "T" or line[0] == "F":
            line_as_list = line.strip().split(" ")
            positive_literal, var = (line_as_list[0] == "T"), int(line_as_list[3])
            index, vtree_index = int(line_as_list[1]), int(line_as_list[2])
            parameters: List[float] = []
            for i in range(self._num_classes):
                parameters.append(float(line_as_list[4 + i]))
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
                    {-var}
                )
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

        # Read decision nodes, creates both And and Or gates
        root: Optional[OrGate] = None
        while line[0] == "D":
            line_as_list = line.strip().split(" ")
            index, vtree_index, num_elements = int(line_as_list[1]), int(line_as_list[2]), int(line_as_list[3])
            elements: List[AndGate] = []
            variables: Set[int] = set()
            for i in range(num_elements):
                prime_index = int(line_as_list[i * (self._num_classes + 2) + 4].strip("("))
                sub_index = int(line_as_list[i * (self._num_classes + 2) + 5])
                element_variables: Set[int] = nodes[prime_index][1].union(nodes[sub_index][1])
                variables = variables.union(element_variables)
                splittable_variables: Set[int] = set()
                for variable in element_variables:
                    if -variable in element_variables:
                        splittable_variables.add(abs(variable))
                parameters: List[float] = []
                for j in range(self._num_classes):
                    parameters.append(
                        float(line_as_list[i * (self._num_classes + 2) + 6 + j].strip(")")))
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

        del nodes
        gc.collect()
        return root


#
# light wrapper for structure learning


def learn_logistic_circuit(
    vtree: Vtree,
    n_classes: int,
    # train_x, train_y,
    train: DataSet,
    valid: Optional[DataSet] = None,
    solver: str = "auto",
    max_iter_sl: int = 1000,
    max_iter_pl: int = 1000,
    depth: int = 20,
    num_splits: int = 10,
    C: Union[List[float], int] = 10,
    validate_every: int = 10,
    patience: int = 2,
    rand_gen: Optional[RandomState] = None
) -> Tuple[LogisticCircuit, List[float]]:

    # #
    # # FIXEME: do we need this?
    # X[np.where(X == 0.0)[0]] = 1e-5
    # X[np.where(X == 1.0)[0]] -= 1e-5

    # train = Dataset(train_x, train_y)

    accuracy_history: List[float] = []

    circuit = LogisticCircuit(vtree, n_classes, rand_gen=rand_gen)
    train.features = circuit.calculate_features(train.images)

    logging.info(f"The starting circuit has {circuit.num_parameters} parameters.")
    train.features = circuit.calculate_features(train.images)
    train_acc: float = circuit.calculate_accuracy(train)
    logging.info(f" accuracy: {train_acc:.5f}")
    accuracy_history.append(train_acc)

    logging.info("Start structure learning.")
    sl_start_t: float = perf_counter()

    valid_best = -np.inf
    best_model = copy.deepcopy(circuit)
    c_patience = 0

    for i in range(max_iter_sl):
        cur_time = perf_counter()

        circuit.change_structure(train, depth, num_splits)

        train.features = circuit.calculate_features(train.images)
        pl_start_t = perf_counter()
        circuit.learn_parameters(train, max_iter_pl, C=C, rand_gen=rand_gen, solver=solver)
        pl_end_t = perf_counter()

        train_acc: float = circuit.calculate_accuracy(train)

        logging.info(f"done iter {i+1}/{max_iter_sl} in {perf_counter() - cur_time} secs")
        logging.info(f"\tcircuit size: {circuit.num_parameters}")
        logging.info(f"\tparameters learning done in {pl_end_t - pl_start_t} secs")
        logging.info(f"\taccuracy: {train_acc:.5f} size {circuit.num_parameters}")
        accuracy_history.append(train_acc)

        if i % validate_every == 0 and valid is not None:
            logging.info(f"evaluate in the validation set")
            valid.features = circuit.calculate_features(valid.images)
            valid_acc = circuit.calculate_accuracy(valid)
            if valid_acc <= valid_best:
                logging.info(f"Worsening on valid: {valid_acc} <= prev best {valid_best}")
                if c_patience >= patience:
                    logging.info(f"Exceeding patience {c_patience} >= {patience}: STOP")
                    break
                else:
                    c_patience += 1
            else:
                logging.info(f'Found new best model {valid_acc}')
                best_model = copy.deepcopy(circuit)
                valid_best = valid_acc
                c_patience = 0

    sl_end_t: float = perf_counter()
    logging.info(f"Structure learning done in {sl_end_t - sl_start_t} secs")

    return best_model, accuracy_history
