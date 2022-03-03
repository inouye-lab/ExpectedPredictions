from typing import Dict, List, Tuple, Optional, NoReturn

from pypsdd.psdd import PSddNode
from LogisticCircuit.structure.CircuitNode import CircuitNode as LogisticCircuitNode

import numpy as np

PCIndex = PSddNode
LCIndex = LogisticCircuitNode
EVIndex = Tuple[PCIndex, LCIndex]
MIndex = int
MomentIndex = Tuple[MIndex, PCIndex, LCIndex]
CacheData = np.ndarray


def psdd_index(psdd: PSddNode) -> PCIndex:
    """Creates a cache index from teh given PSDD node"""
    return psdd  # TODO


def lgc_index(lgc: LogisticCircuitNode) -> LCIndex:
    """Creates a cache index from teh given LC node"""
    return lgc


class EVCache:
    f_cache: Dict[EVIndex, CacheData]
    g_cache: Dict[EVIndex, CacheData]
    fg_cache: Dict[EVIndex, CacheData]

    exp_order: List[Tuple[str, PCIndex, LCIndex]]

    moment_g_cache: Dict[MomentIndex, CacheData]
    moment_fg_cache: Dict[MomentIndex, CacheData]

    def __init__(self):
        self.f_cache = dict()
        self.g_cache = dict()
        self.fg_cache = dict()

        self.exp_order = []

        self.moment_g_cache = dict()
        self.moment_fg_cache = dict()

    def get_f(self, psdd_id: PCIndex, lgc_id: LCIndex) -> CacheData:
        return self._get("f", self.f_cache, psdd_id, lgc_id)

    def put_f(self, psdd_id: PCIndex, lgc_id: LCIndex, value: CacheData) -> NoReturn:
        self._put(self.f_cache, psdd_id, lgc_id, value)

    def get_g(self, psdd_id: PCIndex, lgc_id: LCIndex) -> CacheData:
        return self._get("g", self.g_cache, psdd_id, lgc_id)

    def put_g(self, psdd_id: PCIndex, lgc_id: LCIndex, value: CacheData) -> NoReturn:
        self._put(self.g_cache, psdd_id, lgc_id, value)

    def get_fg(self, psdd_id: PCIndex, lgc_id: LCIndex) -> CacheData:
        return self._get("fg", self.fg_cache, psdd_id, lgc_id)

    def put_fg(self, psdd_id: PCIndex, lgc_id: LCIndex, value) -> NoReturn:
        self._put(self.fg_cache, psdd_id, lgc_id, value)

    def get_moment_g(self, psdd_id: PCIndex, lgc_id: LCIndex, moment: MIndex):
        if (moment, psdd_id, lgc_id) in self.moment_g_cache:
            return self.moment_g_cache[(moment, psdd_id, lgc_id)]
        return None

    def put_moment_g(self, psdd_id: PCIndex, lgc_id: LCIndex, moment: MIndex, value: CacheData) -> NoReturn:
        self.moment_g_cache[(moment, psdd_id, lgc_id)] = value

    def get_moment_fg(self, psdd_id: PCIndex, lgc_id: LCIndex, moment: MIndex):
        if (moment, psdd_id, lgc_id) in self.moment_fg_cache:
            return self.moment_fg_cache[(moment, psdd_id, lgc_id)]
        return None

    def put_moment_fg(self, psdd_id: PCIndex, lgc_id: LCIndex, moment: MIndex, value: CacheData) -> NoReturn:
        self.moment_fg_cache[(moment, psdd_id, lgc_id)] = value

    ###### 
    def _get(self, types: str, cache: Dict[EVIndex, CacheData], psdd_id: PCIndex, lgc_id: LCIndex) -> Optional[CacheData]:
        if (psdd_id, lgc_id) in cache:
            return cache[(psdd_id, lgc_id)]
        else:
            self.exp_order.append((types, psdd_id, lgc_id))
            return None

    @staticmethod
    def _put(cache: Dict[EVIndex, CacheData], psdd_id: PCIndex, lgc_id: LCIndex, value: CacheData) -> NoReturn:
        cache[(psdd_id, lgc_id)] = value

    ######

    def clear(self):
        self.f_cache.clear()
        self.g_cache.clear()
        self.fg_cache.clear()
        self.moment_g_cache.clear()
        self.moment_fg_cache.clear()