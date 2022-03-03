from typing import Union, Optional, Set, NoReturn

import numpy as np

from .CircuitNode import CircuitNode, OrGate, CircuitTerminal


"""Simple union allowing us to assume if something is not a terminal, it is an or gate"""
AndChildNode = Union[CircuitTerminal, OrGate]


class AndGate(CircuitNode):
    """
    And Gate.
    We also refer AND Gates as Elements.
    In this implementation, we assume every AND gate is the child of one PSDD decision nodes (OR gate).
    In another words, they are not shared between different PSDD decision nodes.
    """

    _prime: Optional[AndChildNode]
    _sub: Optional[AndChildNode]
    _parameter: Optional[np.ndarray]
    _splittable_variables: Set[int]
    _parent: Optional[OrGate]
    _flag: bool

    def __init__(self, prime: Optional[AndChildNode], sub: Optional[AndChildNode],
                 parameter: Optional[np.ndarray] = None):
        # TODO: super call
        # super().__init__(index = None, vtree = None)
        self._prime = prime
        self._sub = sub
        self._prime.increase_num_parents_by_one()
        self._sub.increase_num_parents_by_one()
        # difference between prob and feature:
        # prob is calculated in a bottom-up pass and only considers values of variables the element has
        # feature is calculated in a top-down pass using probs; equals the WMC of that element reached
        self._prob = None
        self._feature = None
        self._parameter = parameter
        self._parent = None
        self._splittable_variables = set()
        self._flag = False

    def __repr__(self):
        return "(" + self._prime.__repr__() + ", " + self._sub.__repr__() + ")"

    @property
    def prime(self) -> Optional[AndChildNode]:
        return self._prime

    @prime.setter
    def prime(self, value: Optional[AndChildNode]):
        self._prime = value
        if self._prime is not None:
            self._prime.increase_num_parents_by_one()

    @property
    def sub(self) -> Optional[AndChildNode]:
        return self._sub

    @sub.setter
    def sub(self, value: Optional[AndChildNode]) -> NoReturn:
        self._sub = value
        if self._sub is not None:
            self._sub.increase_num_parents_by_one()

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, value) -> NoReturn:
        self._feature = value

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, value) -> NoReturn:
        self._prob = value

    def calculate_prob(self) -> NoReturn:
        self._prob = self._prime.prob + self._sub.prob

    @property
    def parameter(self) -> Optional[np.ndarray]:
        return self._parameter

    @parameter.setter
    def parameter(self, value: Optional[np.ndarray]):
        self._parameter = value

    @property
    def parent(self) -> Optional[OrGate]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional[OrGate]) -> NoReturn:
        self._parent = value

    @property
    def splittable_variables(self) -> Set[int]:
        return self._splittable_variables

    @splittable_variables.setter
    def splittable_variables(self, value: Set[int]):
        self._splittable_variables = value

    def remove_splittable_variable(self, variable_to_remove: int):
        self._splittable_variables.discard(variable_to_remove)

    @property
    def flag(self) -> bool:
        return self._flag

    @flag.setter
    def flag(self, value: bool) -> NoReturn:
        self._flag = value
