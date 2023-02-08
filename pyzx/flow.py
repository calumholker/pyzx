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

from typing import Dict, Set, Tuple, Optional
from math import comb

from .linalg import Mat2
from .graph.base import BaseGraph, VertexType, VT, ET


def gflow(
    g: BaseGraph[VT, ET]
) -> Optional[Tuple[Dict[VT, int], Dict[VT, Set[VT]], int]]:
    """Compute the maximally delayed gflow of a diagram in graph-like form.

    Based on algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70
    """
    l: Dict[VT, int] = {}
    gflow: Dict[VT, Set[VT]] = {}

    inputs: Set[VT] = set(g.inputs())
    processed: Set[VT] = set(g.outputs()) | g.grounds()
    vertices: Set[VT] = set(g.vertices())
    pattern_inputs: Set[VT] = set()
    for inp in inputs:
        if g.type(inp) == VertexType.BOUNDARY:
            pattern_inputs |= set(g.neighbors(inp))
        else:
            pattern_inputs.add(inp)
    k: int = 1

    for v in processed:
        l[v] = 0

    while True:
        correct = set()
        # unprocessed = list()
        processed_prime = [
            v
            for v in processed.difference(pattern_inputs)
            if any(w not in processed for w in g.neighbors(v))
        ]
        candidates = [
            v
            for v in vertices.difference(processed)
            if any(w in processed_prime for w in g.neighbors(v))
        ]

        zerovec = Mat2([[0] for i in range(len(candidates))])
        # print(unprocessed, processed_prime, zerovec)
        m = Mat2([[1 if g.connected(v,w) else 0 for v in processed_prime] for w in candidates])
        for u in candidates:
            vu = zerovec.copy()
            vu.data[candidates.index(u)] = [1]
            x = m.solve(vu)
            if x:
                correct.add(u)
                gflow[u] = {processed_prime[i] for i in range(x.rows()) if x.data[i][0]}
                l[u] = k

        if not correct:
            if not candidates:
                return l, gflow, k
            return None
        else:
            processed.update(correct)
            k += 1

def flow(
  g: BaseGraph[VT, ET]
) -> Optional[Tuple[Dict[VT, int], Dict[VT, Set[VT]], int]]:
    """Compute the causal flow of a diagram in graph-like form.
    Computes in O(kn) for vertices n and k=|I|=|O| 

    Based on algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70
    """
    order: Dict[VT, int] = {}
    flow: Dict[VT, VT] = {}
    
    inputs: Set[VT] = set(g.inputs())
    processed: Set[VT] = set(g.outputs())
    vertices: Set[VT] = set(g.vertices())
    correctors = processed.difference(inputs)
    k: int = 1
    
    for v in processed:
        order[v] = 0

    while True:
        Out_prime = set()
        C_prime = set()
        
        for v in correctors:
            ns = set(g.neighbors(v)).difference(processed)
            if len(ns) == 1 and v not in ns:
                (u,) = ns
                flow[u] = v
                order[v] = k
                Out_prime |= {u}
                C_prime |= {v}

        if not Out_prime:
            if processed == vertices:
                return order, flow, k
            return None
        else:
            processed |= Out_prime
            correctors = (correctors.difference(C_prime)) | (Out_prime.intersection(vertices.difference(inputs)))
            k += 1
