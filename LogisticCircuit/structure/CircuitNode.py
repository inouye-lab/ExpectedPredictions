import numpy as np
import torch

from .Vtree import Vtree
from typing import List, NoReturn, Optional, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .AndGate import AndGate


class CircuitNode(object):
    _index: int
    _vtree: Vtree
    _num_parents: int
    _prob: Optional[np.ndarray]
    _feature: Optional[np.ndarray]

    def __init__(self, index: int, vtree: Vtree):
        self._index = index
        self._vtree = vtree
        self._num_parents = 0
        # difference between prob and feature:
        # prob is calculated in a bottom-up pass and only considers values of variables the node has
        # feature is calculated in a top-down pass using probs; equals the WMC of that node reached
        self._prob = None
        self._feature = None

    @property
    def index(self) -> int:
        return self._index

    @property
    def vtree(self) -> Vtree:
        return self._vtree

    @property
    def num_parents(self) -> int:
        return self._num_parents

    def increase_num_parents_by_one(self) -> NoReturn:
        self._num_parents += 1

    def decrease_num_parents_by_one(self) -> NoReturn:
        self._num_parents -= 1

    @property
    def feature(self) -> Optional[np.ndarray]:
        return self._feature

    @feature.setter
    def feature(self, value: Optional[np.ndarray]) -> NoReturn:
        self._feature = value

    @property
    def prob(self) -> Optional[np.ndarray]:
        return self._prob

    @prob.setter
    def prob(self, value: Optional[np.ndarray]) -> NoReturn:
        self._prob = value

    @staticmethod
    def _safe_log(arr: np.ndarray) -> np.ndarray:
        log_values = np.zeros(arr.shape)
        LOG_ZERO = -10000
        log_values.fill(LOG_ZERO)
        non_zero_ids = arr != 0
        log_values[non_zero_ids] = np.log(arr[non_zero_ids])
        return log_values


class OrGate(CircuitNode):
    """OR Gate.
       Or gates are also referred as Decision nodes."""

    _elements: List['AndGate']

    def __init__(self, index: int, vtree: Vtree, elements: List['AndGate']):
        super().__init__(index, vtree)
        self._elements = elements
        for element in elements:
            element.parent = self

    def __repr__(self) -> str:
        return "D {}".format(self._index)

    @property
    def elements(self) -> List['AndGate']:
        return self._elements

    def add_element(self, element: 'AndGate') -> NoReturn:
        self._elements.append(element)
        element.parent = self

    def remove_element(self, index: int) -> NoReturn:
        del self._elements[index]

    def calculate_prob(self) -> NoReturn:
        if len(self._elements) == 0:
            raise ValueError("Decision nodes should have at least one elements.")
        for element in self._elements:
            element.calculate_prob()
        self._prob = np.sum([np.exp(element.prob) for element in self._elements], axis=0)
        # self._prob = np.where(self._prob < 1e-5, 1e-5, self._prob)
        self._prob = CircuitNode._safe_log(self._prob)
        for element in self._elements:
            element.prob -= self._prob
        self._prob = np.where(self._prob > 0.0, 0.0, self._prob)
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float64)

    def calculate_feature(self) -> NoReturn:
        feature = CircuitNode._safe_log(self._feature)
        for element in self._elements:
            element.feature = np.exp(feature + element.prob)
            element.prime.feature += element.feature
            element.sub.feature += element.feature

    def save(self, f: TextIO) -> NoReturn:
        f.write(f"D {self._index} {self._vtree.index} {len(self._elements)}")
        for element in self._elements:
            f.write(f" ({element.prime.index} {element.sub.index}")
            for parameter in element.parameter:
                f.write(f" {parameter}")
            f.write(f")")
        f.write("\n")


LITERAL_IS_TRUE: int = 1
LITERAL_IS_FALSE: int = 0
LITERAL_IS_TAUTOLOGY: int = 2


class CircuitTerminal(CircuitNode):
    """Terminal(leaf) node."""

    _var_index: int
    _var_value: int
    _parameter: torch.Tensor
    _feature: np.ndarray
    _prob: np.ndarray

    def __init__(self, index: int, vtree: Vtree, var_index: int, var_value: int, parameter: torch.Tensor = None):
        super().__init__(index, vtree)
        self._var_index = var_index
        self._var_value = var_value
        self._parameter = parameter

    def __repr__(self) -> str:
        if self.var_value:
            return "T {}".format(self._var_index)
        else:
            return "F {}".format(self._var_index)

    @property
    def var_index(self) -> int:
        return self._var_index

    @var_index.setter
    def var_index(self, value: int) -> NoReturn:
        self._var_index = value

    @property
    def var_value(self) -> int:
        return self._var_value

    @var_value.setter
    def var_value(self, value: int) -> NoReturn:
        self._var_value = value

    @property
    def parameter(self) -> torch.Tensor:
        return self._parameter

    @parameter.setter
    def parameter(self, value: torch.Tensor) -> NoReturn:
        self._parameter = value

    def calculate_prob(self, samples: np.array) -> NoReturn:
        if self._var_value == LITERAL_IS_TRUE:
            self._prob = CircuitNode._safe_log(samples[:, self._var_index - 1])
        elif self._var_value == LITERAL_IS_FALSE:
            self._prob = CircuitNode._safe_log(1.0 - samples[:, self._var_index - 1])
        else:
            self._prob = CircuitNode._safe_log(np.zeros(len(samples)))
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float64)

    def save(self, f: TextIO) -> NoReturn:
        if self._var_value == LITERAL_IS_TRUE:
            f.write(f"T {self._index} {self._vtree.index} {self._var_index}")
        elif self._var_value == LITERAL_IS_FALSE:
            f.write(f"F {self._index} {self._vtree.index} {self._var_index}")
        else:
            f.write(f"S {self._index} {self._vtree.index} {self._var_index}")
        for parameter in self._parameter:
            f.write(f" {parameter}")
        f.write("\n")
