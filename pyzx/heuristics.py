# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, List, Dict, Set, FrozenSet
from typing import Any, Callable, TypeVar, Optional, Union
from typing_extensions import Literal

from fractions import Fraction
import math

from .utils import VertexType, EdgeType, toggle_edge, vertex_is_zx, FloatInt, FractionLike
from .graph.base import BaseGraph, VT, ET

MatchPivotUnfuseType = Tuple[VT,VT,List[VT],List[VT],Tuple[VT,VT]]
MatchLcompUnfuseType = Tuple[VT,List[VT],VT]
MatchIdFuseType = Tuple[VT,VT,VT]

def pivot_statistics(g: BaseGraph[VT,ET], v0: VT, v1: VT, v0b, v1b, neighbours_to_unfuse: Tuple[VT,VT]):
    v0n = set(g.neighbors(v0))
    v0n.remove(v1)
    if len(neighbours_to_unfuse[0]) > 0: 
        v0n = v0n.difference(set(neighbours_to_unfuse[0]))
        num_v0n = len(v0n) + 1
    else: num_v0n = len(v0n)
    
    v1n = set(g.neighbors(v1))
    v1n.remove(v0)
    if len(neighbours_to_unfuse[1]) > 0: 
        v1n = v1n.difference(set(neighbours_to_unfuse[1]))
        num_v1n = len(v1n) + 1
    else: num_v1n = len(v1n)
    
    shared_n = v0n.intersection(v1n)
    num_shared_n = len(shared_n)
    
    v0n.difference_update(shared_n)
    num_v0n -= num_shared_n
    v1n.difference_update(shared_n)
    num_v1n -= num_shared_n
    
    max_new_connections = num_v0n * num_v1n + num_v0n * num_shared_n + num_v1n * num_shared_n
    
    num_edges_between_neighbours = 0
    for v in v0n:
        for n in g.neighbors(v):
            if n in v1n or n in shared_n: num_edges_between_neighbours += 1
    for v in v1n:
        for n in g.neighbors(v):
            if n in shared_n: num_edges_between_neighbours += 1
            
    num_unfusions = sum(len(unfusion) > 0 for unfusion in neighbours_to_unfuse)
    edges_removed = 2*num_edges_between_neighbours - max_new_connections + len(g.neighbors(v0)) + len(g.neighbors(v1)) - 1 - (2*num_unfusions)
    vertices_removed = 2 - (2*num_unfusions)
    return edges_removed, vertices_removed

def lcomp_statistics(g: BaseGraph[VT,ET], v: VT, vn, neighbours_to_unfuse: VT):
    if len(neighbours_to_unfuse) > 0:
        vn = set(vn).difference(set(neighbours_to_unfuse))
        num_vn = len(vn) + 1
    else:
        vn = set(vn)
        num_vn = len(vn)
    
    max_new_connections = num_vn * (num_vn-1)/2
    num_edges_between_neighbours = 0
    for n in vn: num_edges_between_neighbours += len(vn.intersection(set(g.neighbors(n)))) #double counted edges so don't need to multiply by 2
    
    num_unfusions = 0 if len(neighbours_to_unfuse) == 0 else 1
    edges_removed = num_edges_between_neighbours - max_new_connections + num_vn - (2*num_unfusions)
    vertices_removed = 1 - (2*num_unfusions)
    return edges_removed, vertices_removed

def id_fuse_statistics(g: BaseGraph[VT,ET], v: VT, v0: VT, v1: VT):
    edges_removed = 2
    vertices_removed = 2
    for n in g.neighbors(v0):
        if n == v: continue
        if n in g.neighbors(v1):
            if g.edge_type(g.edge(n,v0)) == g.edge_type(g.edge(n,v1)):
                edges_removed += 2
                if len(g.neighbors(n)) == 2: vertices_removed += 1
            else: edges_removed += 1   
    if g.connected(v0,v1): edges_removed += 1
    return edges_removed, vertices_removed

def lcomp_2Q_simp_heuristic(g: BaseGraph[VT,ET], match: MatchLcompUnfuseType, score_params):
    edges_removed, vertices_removed = lcomp_statistics(g, match[0], match[1], match[2])
    twoQ_removed = edges_removed - vertices_removed
    if twoQ_removed > 0: return score_params[0]*twoQ_removed
    if twoQ_removed == 0 and vertices_removed > 0: return score_params[0]*twoQ_removed
    return None

def pivot_2Q_simp_heuristic(g: BaseGraph[VT,ET], match: MatchPivotUnfuseType, score_params):
    edges_removed, vertices_removed = pivot_statistics(g, match[0], match[1], match[2], match[3], match[4])
    twoQ_removed = edges_removed - vertices_removed
    if twoQ_removed > 0: return score_params[1]*twoQ_removed
    if twoQ_removed == 0 and vertices_removed > 0: return score_params[1]*twoQ_removed
    return None

def id_fuse_2Q_reduce_heuristic(g: BaseGraph[VT,ET], match: MatchIdFuseType, score_params):
    edges_removed, vertices_removed = id_fuse_statistics(g, match[0], match[1], match[2])
    twoQ_removed = edges_removed - vertices_removed
    if twoQ_removed >= 0: return score_params[2]*twoQ_removed