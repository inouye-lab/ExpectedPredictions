import copy
import gc
from abc import abstractmethod
from collections import deque

import numpy as np
import torch
from typing import Optional, List, NoReturn, Set, Deque, Tuple, Union, TextIO

from numpy.random import RandomState

from ..structure.AndGate import AndGate, AndChildNode
from ..structure.CircuitNode import CircuitTerminal, OrGate
from ..structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE
from ..structure.Vtree import Vtree


class BaseCircuit(object):
    """Shared base class for logistic and regression circuits"""

    _vtree: Vtree
    _num_classes: int
    _largest_index: int
    _num_variables: int
    rand_gen: RandomState

    _terminal_nodes: List[Optional[CircuitTerminal]]
    _decision_nodes: Optional[List[OrGate]]
    _elements: Optional[List[AndGate]]
    _parameters: Optional[torch.Tensor]
    _covariance: List[np.ndarray]
    _bias: torch.Tensor
    _root: OrGate

    def __init__(self, vtree: Vtree, num_classes: int, circuit_file: Optional[TextIO] = None,
                 rand_gen: Optional[RandomState] = None, requires_grad: bool = False):
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
        self._covariance = []

        self._bias = torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))  # TODO: copy needed?

        if circuit_file is None:
            self._generate_all_terminal_nodes(vtree)
            self._root = self._new_psdd(vtree)
        else:
            self._root = self.load(circuit_file)

        self._serialize()

        # enable torch for parameter gradient processing if requested
        if requires_grad:
            # due to the way the parameters are set up, not much easier way to do this
            # we need _parameters to be the source of the parameters in the node
            self._parameters = self._parameters.clone()
            self._parameters.requires_grad = True
            self.set_node_parameters(self._parameters)
            gc.collect()

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
    def covariance(self) -> List[np.ndarray]:
        return self._covariance

    def zero_grad(self, requires_grad: bool = True):
        """
        Zeroes the gradient vector of the parameters vector
        Note that when computing multiple gradients, its essential to recreate the expectation cache for each input
        If using the same input for multiple gradients, it is nessesscary to retain the graph
        """
        if self._parameters.grad is not None:
            # disable gradients if no longer needed
            if not requires_grad:
                self._parameters.detach_()
            self._parameters.grad.zero_()
        # enforce that the circuit is either ready or not ready for gradients
        self._parameters.requires_grad = requires_grad


    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    @abstractmethod
    def _parameter_size(self) -> Union[int, Tuple[int]]:
        """Gets the size of the parameters used in generation"""
        pass

    def _generate_all_terminal_nodes(self, vtree: Vtree) -> NoReturn:
        if vtree.is_leaf():
            var_index = vtree.var
            self._terminal_nodes[var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_TRUE,
                torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))  # TODO: can we directly generate using torch?
            )
            self._largest_index += 1
            self._terminal_nodes[self._num_variables + var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_FALSE,
                torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))  # TODO: can we directly generate using torch?
            )
            self._largest_index += 1
        else:
            self._generate_all_terminal_nodes(vtree.left)
            self._generate_all_terminal_nodes(vtree.right)

    def _new_psdd(self, vtree: Vtree) -> OrGate:
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
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
        elif left_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._terminal_nodes[prime_variable - 1],
                    self._new_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            elements.append(
                AndGate(
                    self._terminal_nodes[self._num_variables + prime_variable - 1],
                    self._new_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(right_vtree.variables)
        elif right_vtree.is_leaf():
            elements.append(
                AndGate(
                    self._new_psdd(left_vtree),
                    self._terminal_nodes[sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            elements.append(
                AndGate(
                    self._new_psdd(left_vtree),
                    self._terminal_nodes[self._num_variables + sub_variable - 1],
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
                )
            )
            for element in elements:
                element.splittable_variables = copy.deepcopy(left_vtree.variables)
        else:
            elements.append(
                AndGate(
                    self._new_psdd(left_vtree),
                    self._new_psdd(right_vtree),
                    torch.tensor(self.rand_gen.random_sample(size=self._parameter_size())),
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
        decision_node_indices: Set[int] = set()
        decision_node_indices.add(self._root.index)
        unvisited: Deque[OrGate] = deque()
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

    def set_node_parameters(self, parameters: torch.Tensor, set_circuit: bool = False,
                            reset_covariance: bool = False) -> NoReturn:
        """
        Sets the parameters of the nodes from the given parameter tensor.
        Intentionally does not update self.parameters to allow more samples.

        @param parameters:       New parameters to set
        @param set_circuit:      If true, sets the parameters on the circuit itself
        @param reset_covariance: If true, resets the covariance on the circuit
        """
        if set_circuit:
            self._parameters = parameters.clone()
            parameters = self._parameters
        self._bias = parameters[:, 0]
        for i in range(len(self._terminal_nodes)):
            self._terminal_nodes[i].parameter = parameters[:, i + 1]
        for i in range(len(self._elements)):
            self._elements[i].parameter = parameters[:, i + 1 + 2 * self._num_variables]
        if reset_covariance:
            self._covariance = []
        gc.collect()

    def randomize_node_parameters(self) -> NoReturn:
        """Randomizes all the parameters on the circuit, used for retraining to minimize bias"""
        self._bias = torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))
        for i in range(len(self._terminal_nodes)):
            self._terminal_nodes[i].parameter = torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))
        for i in range(len(self._elements)):
            self._elements[i].parameter = torch.tensor(self.rand_gen.random_sample(size=self._parameter_size()))
        self._serialize()

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

    # _select_element_and_variable_to_split

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

    @abstractmethod
    def save(self, f: TextIO) -> NoReturn:
        pass

    @abstractmethod
    def load(self, f: TextIO) -> OrGate:
        pass
