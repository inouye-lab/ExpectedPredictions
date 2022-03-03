from collections import deque
from abc import ABC, abstractmethod
from typing import Optional, Set, Deque, List, TextIO, Union, Tuple
import numpy as np
from numpy.random import RandomState

VTREE_FORMAT = """c ids of vtree nodes start at 0
c ids of variables start at 1
c vtree nodes appear bottom-up, children before parents
c
c file syntax:
c vtree number-of-nodes-in-vtree
c L id-of-leaf-vtree-node id-of-variable
c I id-of-internal-vtree-node id-of-left-child id-of-right-child
c
"""


class Vtree(ABC):
    _index: int
    _var_count: int

    def __init__(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def var_count(self) -> int:
        return self._var_count

    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @property
    def left(self) -> Optional['Vtree']:
        return None

    @property
    def right(self) -> Optional['Vtree']:
        return None

    @property
    def var(self) -> int:
        return 0

    @property
    @abstractmethod
    def variables(self) -> Set[int]:
        pass

    def bfs(self) -> List['Vtree']:
        visited: List['Vtree'] = []
        nodes_to_visit: Deque['Vtree'] = deque()
        nodes_to_visit.append(self)
        while nodes_to_visit:
            n = nodes_to_visit.popleft()
            visited.append(n)
            if not n.is_leaf():
                nodes_to_visit.append(n.left)
                nodes_to_visit.append(n.right)

        return visited

    @staticmethod
    def read(file) -> 'Vtree':
        with open(file, 'r') as vtree_file:
            vtree_file: TextIO
            line = 'c'
            while line[0] == 'c':
                line = vtree_file.readline()
            if line.strip().split(' ')[0] != 'vtree':
                raise ValueError('Number of vtree nodes is not specified')
            num_nodes = int(line.strip().split(' ')[1])
            nodes: List[Optional[Vtree]] = [None] * num_nodes
            root: Optional[Vtree] = None
            for line in vtree_file.readlines():
                line_as_list: List[str] = line.strip().split(' ')
                index = int(line_as_list[1])
                if line_as_list[0] == 'L':
                    root = VtreeLeaf(index, int(line_as_list[2]))
                elif line_as_list[0] == 'I':
                    root = VtreeIntermediate(index, nodes[int(line_as_list[2])], nodes[int(line_as_list[3])])
                else:
                    raise ValueError('Vtree node could only be L or I')
                nodes[index] = root
            if root is None:
                raise ValueError('Vtree has no elements')
            return root

    def save(self, file):
        leaves_before_parents: List[Vtree] = list(reversed(self.bfs()))
        n_nodes: int = len(leaves_before_parents)
        print('There are ', n_nodes)
        with open(file, 'w') as f_out:
            f_out: TextIO
            f_out.write(VTREE_FORMAT)
            f_out.write(f'vtree {n_nodes}\n')

            for n in leaves_before_parents:
                if isinstance(n, VtreeLeaf):
                    f_out.write(f'L {n.index} {n.var}\n')
                elif isinstance(n, VtreeIntermediate):
                    f_out.write(f'I {n.index} {n.left.index} {n.right.index}\n')
                else:
                    raise ValueError('Unrecognized vtree node type', n)


class VtreeLeaf(Vtree):
    _var: int

    def __init__(self, index: int, variable: int):
        super(VtreeLeaf, self).__init__(index)
        self._var = variable
        self._var_count = 1

    def is_leaf(self) -> bool:
        return True

    @property
    def var(self) -> int:
        return self._var

    @property
    def variables(self) -> Set[int]:
        return {self._var}


class VtreeIntermediate(Vtree):
    _left: Vtree
    _right: Vtree
    _variables: Set[int]

    def __init__(self, index, left, right):
        super(VtreeIntermediate, self).__init__(index)
        self._left = left
        self._right = right
        self._variables = set()
        self._var_count = self._left.var_count + self._right.var_count
        self._variables.update(self._left.variables)
        self._variables.update(self._right.variables)

    def is_leaf(self) -> bool:
        return False

    @property
    def left(self) -> Vtree:
        return self._left

    @property
    def right(self) -> Vtree:
        return self._right

    @property
    def variables(self) -> Set[int]:
        return self._variables


#
# Generate vtrees
#
RAND_SEED = 1337


def balanced_random_split(variables: np.ndarray, index: int, rand_gen: RandomState) -> Tuple[Vtree, int]:
    n_vars = len(variables)

    if n_vars > 1:
        rand_gen.shuffle(variables)
        mid = n_vars // 2
        var_left, var_right = variables[:mid], variables[mid:]

        node_left, id_left = balanced_random_split(var_left, index, rand_gen)
        node_right, id_right = balanced_random_split(var_right, id_left, rand_gen)

        v = VtreeIntermediate(id_right, node_left, node_right)
        return v, id_right + 1
    else:
        v = VtreeLeaf(index, variables[0])
        return v, index + 1


def unbalanced_random_split(variables: np.ndarray, index: int, rand_gen: RandomState,
                            beta_prior: Tuple[float, float] = (0.3, 0.3)) -> Tuple[Vtree, int]:
    n_vars = len(variables)

    if n_vars > 1:
        rand_gen.shuffle(variables)

        rand_split = rand_gen.beta(a=beta_prior[0], b=beta_prior[1])
        mid = max(min(int(rand_split * n_vars), n_vars - 1), 1)
        var_left, var_right = variables[:mid], variables[mid:]

        node_left, id_left = unbalanced_random_split(var_left, index, rand_gen, beta_prior)
        node_right, id_right = unbalanced_random_split(var_right, id_left, rand_gen, beta_prior)

        v = VtreeIntermediate(id_right, node_left, node_right)
        return v, id_right + 1
    else:
        v = VtreeLeaf(index, variables[0])
        return v, index + 1


def generate_random_vtree(n_vars: int, rand_gen: Union[RandomState, None] = None, balanced: bool = True,
                          beta_prior: Tuple[float, float] = (0.3, 0.3)) -> Vtree:

    if rand_gen is None:
        rand_gen = np.random.RandomState(RAND_SEED)

    variables: np.ndarray = np.arange(n_vars) + 1
    if balanced:
        v, _ = balanced_random_split(variables, index=0, rand_gen=rand_gen)
    else:
        v, _ = unbalanced_random_split(variables, index=0, rand_gen=rand_gen, beta_prior=beta_prior)

    return v
