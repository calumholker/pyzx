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

"""
This module contains the implementation of all the rewrite rules on ZX-diagrams in PyZX.

Each rewrite rule consists of two methods: a matcher and a rewriter.
The matcher finds as many non-overlapping places where the rewrite rule can be applied.
The rewriter takes in a list of matches, and performs the necessary changes on the graph to implement the rewrite.

Each match function takes as input a Graph instance, 
and an optional "filter function" that tells the matcher to only consider
the vertices or edges that the filter function accepts.
It outputs a list of "match" objects. What these objects look like differs
per rewrite rule.

The rewrite function takes as input a Graph instance and a list of match objects
of the appropriate type. It outputs a 4-tuple 
(edges to add, vertices to remove, edges to remove, isolated vertices check).
The first of these should be fed to :meth:`~pyzx.graph.base.BaseGraph.add_edge_table`,
while the second and third should be fed to
:meth:`~graph.base.BaseGraph.remove_vertices` and :meth:`~pyzx.graph.base.BaseGraph.remove_edges`.
The last parameter is a Boolean that when true means that the rewrite rule can introduce
isolated vertices that should be removed by
:meth:`~pyzx.graph.base.BaseGraph.remove_isolated_vertices`\ .

Dealing with this output is done using either :func:`apply_rule` or :func:`pyzx.simplify.simp`.

Warning:
    There is no guarantee that the matcher does not affect the graph, and currently some matchers
    do in fact change the graph. Similarly, the rewrite function also changes the graph other 
    than through the output it generates (for instance by adding vertices or changes phases).

"""

from typing import Tuple, List, Dict, Set, FrozenSet
from typing import Any, Callable, TypeVar, Optional, Union
from typing_extensions import Literal

from fractions import Fraction
import itertools

from .utils import VertexType, EdgeType, toggle_edge, vertex_is_zx, FloatInt, FractionLike
from .graph.base import BaseGraph, VT, ET
from .heuristics import *

RewriteOutputType = Tuple[Dict[ET,List[int]], List[VT], List[ET], bool]
MatchObject = TypeVar('MatchObject')

def apply_rule(
        g: BaseGraph[VT,ET], 
        rewrite: Callable[[BaseGraph[VT,ET], List[MatchObject]],RewriteOutputType[ET,VT]],
        m: List[MatchObject], 
        check_isolated_vertices:bool=True
        ) -> None:
    etab, rem_verts, rem_edges, check_isolated_vertices = rewrite(g, m)
    g.add_edge_table(etab)
    g.remove_edges(rem_edges)
    g.remove_vertices(rem_verts)
    if check_isolated_vertices: g.remove_isolated_vertices()

MatchBialgType = Tuple[VT,VT,List[VT],List[VT]]

def match_bialg(g: BaseGraph[VT,ET]) -> List[MatchBialgType[VT]]:
    """Does the same as :func:`match_bialg_parallel` but with ``num=1``."""
    return match_bialg_parallel(g, num=1)

#TODO: make it be hadamard edge aware
def match_bialg_parallel(
        g: BaseGraph[VT,ET], 
        matchf:Optional[Callable[[ET],bool]]=None, 
        num: int=-1
        ) -> List[MatchBialgType[VT]]:
    """Finds noninteracting matchings of the bialgebra rule.
    
    :param g: An instance of a ZX-graph.
    :param matchf: An optional filtering function for candidate edge, should
       return True if a edge should considered as a match. Passing None will
       consider all edges.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :rtype: List of 4-tuples ``(v1, v2, neighbors_of_v1,neighbors_of_v2)``
    """
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    phases = g.phases()
    types = g.types()
    
    i = 0
    m = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v0, v1 = g.edge_st(candidates.pop())
        if g.is_ground(v0) or g.is_ground(v1):
            continue
        v0t = types[v0]
        v1t = types[v1]
        v0p = phases[v0]
        v1p = phases[v1]
        if (v0p == 0 and v1p == 0 and
        ((v0t == VertexType.Z and v1t == VertexType.X) or (v0t == VertexType.X and v1t == VertexType.Z))):
            v0n = [n for n in g.neighbors(v0) if not n == v1]
            v1n = [n for n in g.neighbors(v1) if not n == v0]
            if (
                all([types[n] == v1t and phases[n] == 0 for n in v0n]) and
                all([types[n] == v0t and phases[n] == 0 for n in v1n])):
                i += 1
                for v in v0n:
                    for c in g.incident_edges(v): candidates.discard(c)
                for v in v1n:
                    for c in g.incident_edges(v): candidates.discard(c)
                m.append((v0,v1,v0n,v1n))
    return m

def bialg(g: BaseGraph[VT,ET], matches: List[MatchBialgType[VT]]) -> RewriteOutputType[ET,VT]:
    """Performs a certain type of bialgebra rewrite given matchings supplied by
    ``match_bialg(_parallel)``."""
    rem_verts = []
    etab: Dict[ET, List[int]] = dict()
    for m in matches:
        rem_verts.append(m[0])
        rem_verts.append(m[1])
        es = [g.edge(i,j) for i in m[2] for j in m[3]]
        for e in es:
            if e in etab: etab[e][0] += 1
            else: etab[e] = [1,0]
    
    return (etab, rem_verts, [], True)


MatchSpiderType = Tuple[VT,VT]

def match_spider(g: BaseGraph[VT,ET]) -> List[MatchSpiderType[VT]]:
    """Does the same as :func:`match_spider_parallel` but with ``num=1``."""
    return match_spider_parallel(g, num=1)

def match_spider_parallel(
        g: BaseGraph[VT,ET], 
        matchf: Optional[Callable[[ET],bool]] = None, 
        num: int = -1,
        allow_interacting_matches: bool = False
        ) -> List[MatchSpiderType[VT]]:
    """Finds non-interacting matchings of the spider fusion rule.
    
    :param g: An instance of a ZX-graph.
    :param matchf: An optional filtering function for candidate edge, should
       return True if the edge should be considered for matchings. Passing None will
       consider all edges.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once
    :rtype: List of 2-tuples ``(v1, v2)``
    """
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    
    i = 0
    m: List[MatchSpiderType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if g.edge_type(e) != EdgeType.SIMPLE: continue
        
        v0, v1 = g.edge_st(e)
        if not (g.type(v0) == g.type(v1) and vertex_is_zx(g.type(v0))): continue
        
        m.append((v0,v1))
        i += 1
        
        if allow_interacting_matches: continue
        for n in g.neighbors(v0):
            for ne in g.incident_edges(n): candidates.discard(ne)
        for n in g.neighbors(v1):
            for ne in g.incident_edges(n): candidates.discard(ne)
            
    return m

def spider(g: BaseGraph[VT,ET], matches: List[MatchSpiderType[VT]]) -> RewriteOutputType[ET,VT]:
    '''Performs spider fusion given a list of matchings from ``match_spider(_parallel)``
    '''
    rem_verts = []
    etab: Dict[ET,List[int]] = dict()

    for v0, v1 in matches:
        if g.row(v0) == 0: v0, v1 = v1, v0

        if g.is_ground(v0) or g.is_ground(v1):
            g.set_phase(v0, 0)
            g.set_ground(v0)
        else: g.add_to_phase(v0, g.phase(v1))

        if g.simplifier: g.fuse_phases(v0,v1)

        rem_verts.append(v1) # always delete the second vertex in the match

        for n in g.neighbors(v1): # edges from the second vertex are transferred to the first
            if v0 == n: continue
            e = g.edge(v0,n)
            if e not in etab: etab[e] = [0,0]
            etab[e][g.edge_type(g.edge(v1,n))-1] += 1
            
    return (etab, rem_verts, [], True)

def unspider(g: BaseGraph[VT,ET], m: List[Any], qubit:FloatInt=-1, row:FloatInt=-1) -> VT:
    """Undoes a single spider fusion, given a match ``m``. A match is a list with 3
    elements given by::

      m[0] : a vertex to unspider
      m[1] : the neighbors of the new node, which should be a subset of the
             neighbors of m[0]
      m[2] : the phase of the new node. If omitted, the new node gets all of the phase of m[0]

    Returns the index of the new node. Optional parameters ``qubit`` and ``row`` can be used
    to position the new node. If they are omitted, they are set as the same as the old node.
    """
    u = m[0]
    v = g.add_vertex(ty=g.type(u))
    u_is_ground = g.is_ground(u)
    g.set_qubit(v, qubit if qubit != -1 else g.qubit(u))
    g.set_row(v, row if row != -1 else g.row(u))

    g.add_edge(g.edge(u, v))
    for n in m[1]:
        e = g.edge(u,n)
        g.add_edge(g.edge(v,n), edgetype=g.edge_type(e))
        g.remove_edge(e)
    if len(m) >= 3:
        g.add_to_phase(v, m[2])
        if not u_is_ground:
            g.add_to_phase(u, Fraction(0) - m[2])
    else:
        g.set_phase(v, g.phase(u))
        g.set_phase(u, 0)
    return v


MatchPivotType = Tuple[VT,VT,List[VT],List[VT]]

def match_pivot(g: BaseGraph[VT,ET]) -> List[MatchPivotType[VT]]:
    """Does the same as :func:`match_pivot_parallel` but with ``num=1``."""
    return match_pivot_parallel(g, num=1, check_edge_types=True)

def match_pivot_parallel(
        g: BaseGraph[VT,ET], 
        matchf: Optional[Callable[[ET],bool]] = None, 
        num: int = -1, 
        check_edge_types: bool = True,
        allow_interacting_matches: bool = False
        ) -> List[MatchPivotType[VT]]:
    """Finds non-interacting matchings of the pivot rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param matchf: An optional filtering function for candidate edge, should
       return True if a edge should considered as a match. Passing None will
       consider all edges.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once
    :rtype: List of 4-tuples. See :func:`pivot` for the details.
    """
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    
    i = 0
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if check_edge_types and g.edge_type(e) != EdgeType.HADAMARD: continue
        
        v0, v1 = g.edge_st(e)
        if not (g.type(v0) == VertexType.Z and g.type(v1) == VertexType.Z): continue
        if any(g.phase(v) not in (0,1) for v in (v0,v1)): continue
        if g.is_ground(v0) or g.is_ground(v1): continue
        
        invalid_edge = False
        v0n = list(g.neighbors(v0))
        v0b = []
        for n in v0n:
            if g.type(n) == VertexType.Z and g.edge_type(g.edge(v0,n)) == EdgeType.HADAMARD: pass
            elif g.type(n) == VertexType.BOUNDARY: v0b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge: continue

        v1n = list(g.neighbors(v1))
        v1b = []
        for n in v1n:
            if g.type(n) == VertexType.Z and g.edge_type(g.edge(v1,n)) == EdgeType.HADAMARD: pass
            elif g.type(n) == VertexType.BOUNDARY: v1b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge: continue
        if len(v0b) + len(v1b) > 1: continue
        
        m.append((v0,v1,tuple(v0b),tuple(v1b)))
        i += 1
        
        if allow_interacting_matches: continue
        for n in v0n:
            for ne in g.incident_edges(n): candidates.discard(ne)
        for n in v1n:
            for ne in g.incident_edges(n): candidates.discard(ne)
        
    return m

def match_pivot_gadget(
        g: BaseGraph[VT,ET], 
        matchf: Optional[Callable[[ET],bool]] = None, 
        num: int = -1,
        allow_interacting_matches: bool = False
        ) -> List[MatchPivotType[VT]]:
    """Like :func:`match_pivot_parallel`, but except for pairings of
    Pauli vertices, it looks for a pair of an interior Pauli vertex and an
    interior non-Clifford vertex in order to gadgetize the non-Clifford vertex."""
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    
    i = 0
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        v0, v1 = g.edge_st(e)
        if not all(g.type(v) == VertexType.Z for v in (v0,v1)): continue
        
        if g.phase(v0) not in (0,1):
            if g.phase(v1) in (0,1): v0, v1 = v1, v0
            else: continue
        elif g.phase(v1) in (0,1): continue # Now v0 has a Pauli phase and v1 has a non-Pauli phase
        
        if g.is_ground(v0): continue
        
        v0n = list(g.neighbors(v0))
        v1n = list(g.neighbors(v1))
        if len(v1n) == 1: continue # It is a phase gadget
        if any(g.type(n) != VertexType.Z for vn in (v0n,v1n) for n in vn): continue
        
        bad_match = False
        edges_to_discard = []
        for i, neighbors in enumerate((v0n, v1n)):
            for n in neighbors:
                if g.type(n) != VertexType.Z:
                    bad_match = True
                    break
                ne = list(g.incident_edges(n))
                if i == 0 and len(ne) == 1 and not (e == ne[0]): # v0 is a phase gadget
                    bad_match = True
                    break
                edges_to_discard.extend(ne)
            if bad_match: break
        if bad_match: continue
        
        m.append((v0,v1,tuple(),tuple()))
        i += 1
        
        if allow_interacting_matches: continue
        for c in edges_to_discard: candidates.discard(c)
        
    return m

def match_pivot_boundary(
        g: BaseGraph[VT,ET], 
        matchf: Optional[Callable[[VT],bool]] = None, 
        num: int=-1,
        allow_interacting_matches: bool = False
        ) -> List[MatchPivotType[VT]]:
    """Like :func:`match_pivot_parallel`, but except for pairings of
    Pauli vertices, it looks for a pair of an interior Pauli vertex and a
    boundary non-Pauli vertex in order to gadgetize the non-Pauli vertex."""
    if matchf is not None: candidates = set([v for v in g.vertices() if matchf(v)])
    else: candidates = g.vertex_set()
    
    i = 0
    consumed_vertices: Set[VT] = set()
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        if g.type(v) != VertexType.Z or g.phase(v) not in (0,1) or g.is_ground(v): continue
        
        good_vert = True
        w = None
        bound = None
        for n in g.neighbors(v):
            if g.type(n) != VertexType.Z or len(g.neighbors(n)) == 1 or n in consumed_vertices or g.is_ground(n) in consumed_vertices: 
                good_vert = False
                break
            
            boundaries = []
            wrong_match = False
            for b in g.neighbors(n):
                if g.type(b) == VertexType.BOUNDARY: boundaries.append(b)
                elif g.type(b) != VertexType.Z: wrong_match = True
            if len(boundaries) != 1 or wrong_match: continue  # n is not on the boundary or has too many boundaries or has neighbors of wrong type
            if g.phase(n) and hasattr(g.phase(n), 'denominator') and g.phase(n).denominator == 2:
                w = n
                bound = boundaries[0]
            if not w:
                w = n
                bound = boundaries[0]
        if not good_vert or w is None: continue
        assert bound is not None
        
        m.append((v,w,tuple(),tuple([bound])))
        i += 1
        
        if allow_interacting_matches: continue
        for n in g.neighbors(v): 
            consumed_vertices.add(n)
            candidates.discard(n)
        for n in g.neighbors(w): 
            consumed_vertices.add(n)
            candidates.discard(n)
        
    return m

def pivot(g: BaseGraph[VT,ET], matches: List[MatchPivotType[VT]]) -> RewriteOutputType[ET,VT]:
    """Perform a pivoting rewrite, given a list of matches as returned by
    ``match_pivot(_parallel)``. A match is itself a list where:

    ``m[0]`` : first vertex in pivot.
    ``m[1]`` : second vertex in pivot.
    ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
    ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
    """
    rem_verts: List[VT] = []
    rem_edges: List[ET] = []
    etab: Dict[ET,List[int]] = dict()

    for m in matches:
        n = [set(g.neighbors(m[0])), set(g.neighbors(m[1]))]
        for i in range(2):
            n[i].remove(m[1-i]) # type: ignore # Really complex typing situation
            if len(m[i+2]) == 1: n[i].remove(m[i+2][0]) # type: ignore
        
        n.append(n[0] & n[1]) #  n[2] <- non-boundary neighbors of m[0] and m[1]
        n[0] = n[0] - n[2]  #  n[0] <- non-boundary neighbors of m[0] only
        n[1] = n[1] - n[2]  #  n[1] <- non-boundary neighbors of m[1] only
        
        es = ([g.edge(s,t) for s in n[0] for t in n[1]] +
              [g.edge(s,t) for s in n[1] for t in n[2]] +
              [g.edge(s,t) for s in n[0] for t in n[2]])
        k0, k1, k2 = len(n[0]), len(n[1]), len(n[2])
        g.scalar.add_power(k0*k2 + k1*k2 + k0*k1)

        for v in n[2]:
            if not g.is_ground(v): g.add_to_phase(v, 1)

        if g.phase(m[0]) and g.phase(m[1]): g.scalar.add_phase(Fraction(1))
        if not m[2] and not m[3]: g.scalar.add_power(-(k0+k1+2*k2-1))
        elif not m[2]: g.scalar.add_power(-(k1+k2))
        else: g.scalar.add_power(-(k0+k2))

        for i in range(2): # if m[i] has a phase, it will get copied on to the neighbors of m[1-i]:
            a = g.phase(m[i]) # type: ignore
            if a:
                for v in n[1-i]:
                    if not g.is_ground(v): g.add_to_phase(v, a)
                for v in n[2]:
                    if not g.is_ground(v): g.add_to_phase(v, a)

            if not m[i+2]: rem_verts.append(m[1-i]) # if there is no boundary, the other vertex is destroyed
            else:
                e = g.edge(m[i], m[i+2][0]) # type: ignore # if there is a boundary, toggle whether it is an h-edge or a normal edge
                new_e = g.edge(m[1-i], m[i+2][0]) # type: ignore # and point it at the other vertex
                ne,nhe = etab.get(new_e, [0,0])
                if g.edge_type(e) == EdgeType.SIMPLE: nhe += 1
                elif g.edge_type(e) == EdgeType.HADAMARD: ne += 1
                etab[new_e] = [ne,nhe]
                rem_edges.append(e)
            
        for e in es:
            nhe = etab.get(e, (0,0))[1]
            etab[e] = [0,nhe+1]
            
    return (etab, rem_verts, rem_edges, True)

def pivot_gadget(g: BaseGraph[VT,ET], matches: List[MatchPivotType[VT]]) -> RewriteOutputType[ET,VT]:
    """Performs the gadgetizations required before applying pivots.
    ``m[0]`` : interior pauli vertex
    ``m[1]`` : interior non-pauli vertex to gadgetize
    ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
    ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
    """
    vertices_to_gadgetize = [m[1] for m in matches]
    gadgetize(g, vertices_to_gadgetize)
    return pivot(g, matches)

def gadgetize(g: BaseGraph[VT,ET], vertices: List[VT]) -> None:
    """Pulls out a list of vertices into gadgets"""
    edge_list = []
    for v in vertices:
        if any(n in g.inputs() for n in g.neighbors(v)): mod = 0.5
        else: mod = -0.5

        vp = g.add_vertex(VertexType.Z,-2,g.row(v)+mod,g.phase(v))
        v0 = g.add_vertex(VertexType.Z,-1,g.row(v)+mod,0)
        g.set_phase(v, 0)
        
        edge_list.append(g.edge(v,v0))
        edge_list.append(g.edge(v0,vp))
        
        if g.simplifier: g.gadgetize_vertex(vp,v)
        
    g.add_edges(edge_list, EdgeType.HADAMARD)
    return


MatchLcompType = Tuple[VT,List[VT]]

def match_lcomp(g: BaseGraph[VT,ET]) -> List[MatchLcompType[VT]]:
    """Same as :func:`match_lcomp_parallel`, but with ``num=1``"""
    return match_lcomp_parallel(g, num=1, check_edge_types=True)

def match_lcomp_parallel(
        g: BaseGraph[VT,ET], 
        vertexf: Optional[Callable[[VT],bool]] = None, 
        num: int = -1, 
        check_edge_types: bool = True,
        allow_interacting_matches: bool = False
        ) -> List[MatchLcompType[VT]]:
    """Finds noninteracting matchings of the local complementation rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param vertexf: An optional filtering function for candidate vertices, should
       return True if a vertex should be considered as a match. Passing None will
       consider all vertices.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once
    :rtype: List of 2-tuples ``(vertex, neighbors)``.
    """
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    
    i = 0
    m: List[MatchLcompType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        
        if g.type(v) != VertexType.Z: continue
        if g.phase(v) not in (Fraction(1,2), Fraction(3,2)): continue
        if g.is_ground(v): continue

        if check_edge_types and not (
            all(g.edge_type(e) == EdgeType.HADAMARD for e in g.incident_edges(v))
            ): continue
                
        vn = list(g.neighbors(v))
        if any(g.type(n) != VertexType.Z for n in vn): continue
        
        m.append((v,tuple(vn)))
        i += 1
        
        if allow_interacting_matches: continue
        for n in vn: candidates.discard(n)
        
    return m

def lcomp(g: BaseGraph[VT,ET], matches: List[MatchLcompType[VT]]) -> RewriteOutputType[ET,VT]:
    """Performs a local complementation based rewrite rule on the given graph with the
    given ``matches`` returned from ``match_lcomp(_parallel)``. See "Graph Theoretic
    Simplification of Quantum Circuits using the ZX calculus" (arXiv:1902.03178)
    for more details on the rewrite"""
    etab: Dict[ET,List[int]] = dict()
    rem = []
    for v, vn in matches:
        p = g.phase(v)
        rem.append(v)
        
        if p.numerator == 1: g.scalar.add_phase(Fraction(1,4))
        else: g.scalar.add_phase(Fraction(7,4))
        
        n = len(vn)
        g.scalar.add_power((n-2)*(n-1)//2)
        
        for i in range(n):
            if not g.is_ground(vn[i]):
                g.add_to_phase(vn[i], -p)
            for j in range(i+1, n):
                e = g.edge(vn[i],vn[j])
                he = etab.get(e, [0,0])[1]
                etab[e] = [0, he+1]

    return (etab, rem, [], True)


MatchIdType = Tuple[VT,VT,VT,EdgeType.Type]

def match_ids(g: BaseGraph[VT,ET]) -> List[MatchIdType[VT]]:
    """Finds a single identity node. See :func:`match_ids_parallel`."""
    return match_ids_parallel(g, num=1)

def match_ids_parallel(
        g: BaseGraph[VT,ET], 
        vertexf: Optional[Callable[[VT],bool]] = None, 
        num: int = -1,
        allow_interacting_matches: bool = False
        ) -> List[MatchIdType[VT]]:
    """Finds non-interacting identity vertices.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param vertexf: An optional filtering function for candidate vertices, should
       return True if a vertex should be considered as a match. Passing None will
       consider all vertices.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once
    :rtype: List of 4-tuples ``(identity_vertex, neighbor1, neighbor2, edge_type)``.
    """
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()

    i = 0
    m: List[MatchIdType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        if g.phase(v) != 0 or not vertex_is_zx(g.type(v)) or g.is_ground(v): continue
        
        vn = g.neighbors(v)
        if len(vn) != 2: continue
        v0, v1 = vn
        
        if (g.is_ground(v0) and g.type(v1) == VertexType.BOUNDARY or
                g.is_ground(v1) and g.type(v0) == VertexType.BOUNDARY):  # Do not put ground spiders on the boundary
            continue
        
        if g.edge_type(g.edge(v,v0)) != g.edge_type(g.edge(v,v1)): #exactly one of them is a hadamard edge
            m.append((v,v0,v1,EdgeType.HADAMARD))
        else: m.append((v,v0,v1,EdgeType.SIMPLE))
        i += 1
        
        if allow_interacting_matches: continue
        candidates.discard(v0)
        candidates.discard(v1)
        
    return m

def remove_ids(g: BaseGraph[VT,ET], matches: List[MatchIdType[VT]]) -> RewriteOutputType[ET,VT]:
    """Given the output of ``match_ids(_parallel)``, returns a list of edges to add,
    and vertices to remove."""
    etab : Dict[ET,List[int]] = dict()
    rem = []
    for v,v0,v1,et in matches:
        rem.append(v)
        e = g.edge(v0,v1)
        if not e in etab: etab[e] = [0,0]
        if et == EdgeType.SIMPLE: etab[e][0] += 1
        else: etab[e][1] += 1
        
    return (etab, rem, [], False)


MatchGadgetType = Tuple[VT,VT,FractionLike,List[VT],List[VT]]

def match_phase_gadgets(g: BaseGraph[VT,ET]) -> List[MatchGadgetType[VT]]:
    """Determines which phase gadgets act on the same vertices, so that they can be fused together.
    
    :param g: An instance of a ZX-graph.
    :rtype: List of 5-tuples ``(axel,leaf, total combined phase, other axels with same targets, other leafs)``.
    """
    phases = g.phases()

    parities: Dict[FrozenSet[VT], List[VT]] = dict()
    gadgets: Dict[VT,VT] = dict()
    inputs = g.inputs()
    outputs = g.outputs()
    # First we find all the phase-gadgets, and the list of vertices they act on
    for v in g.vertices():
        if phases[v] != 0 and phases[v].denominator > 2 and len(list(g.neighbors(v)))==1:
            n = list(g.neighbors(v))[0]
            if phases[n] not in (0,1): continue # Not a real phase gadget (happens for scalar diagrams)
            if n in gadgets: continue # Not a real phase gadget (happens for scalar diagrams)
            if n in inputs or n in outputs: continue # Not a real phase gadget (happens for non-unitary diagrams)
            gadgets[n] = v
            par = frozenset(set(g.neighbors(n)).difference({v}))
            if par in parities: parities[par].append(n)
            else: parities[par] = [n]

    m: List[MatchGadgetType[VT]] = []
    for par, gad in parities.items():
        if len(gad) == 1: 
            n = gad[0]
            v = gadgets[n]
            if phases[n] != 0: # If the phase of the axel vertex is pi, we change the phase of the gadget
                g.scalar.add_phase(phases[v])
                g.phase_negate(v)
                m.append((v,n,-phases[v],[],[]))
        else:
            totphase = sum((1 if phases[n]==0 else -1)*phases[gadgets[n]] for n in gad)%2
            for n in gad:
                if phases[n] != 0:
                    g.scalar.add_phase(phases[gadgets[n]])
                    g.phase_negate(gadgets[n])
            g.scalar.add_power(-((len(par)-1)*(len(gad)-1)))
            n = gad.pop()
            v = gadgets[n]
            m.append((v,n,totphase, gad, [gadgets[n] for n in gad]))
    return m

def merge_phase_gadgets(g: BaseGraph[VT,ET], matches: List[MatchGadgetType[VT]]) -> RewriteOutputType[ET,VT]:
    """Given the output of :func:``match_phase_gadgets``, removes phase gadgets that act on the same set of targets."""
    rem = []
    for v, n, phase, othergadgets, othertargets in matches:
        g.set_phase(v, phase)
        g.set_phase(n, 0)
        rem.extend(othergadgets)
        rem.extend(othertargets)
        for w in othertargets:
            if g.simplifier:
                g.fuse_phases(v,w)
            if g.merge_vdata is not None:
                g.merge_vdata(v, w)
    return ({}, rem, [], False)


MatchSupplementarityType = Tuple[VT,VT,Literal[1,2],FrozenSet[VT]]

def match_supplementarity(g: BaseGraph[VT,ET]) -> List[MatchSupplementarityType[VT]]:
    """Finds pairs of non-Clifford spiders that are connected to exactly the same set of vertices.
    
    :param g: An instance of a ZX-graph.
    :rtype: List of 4-tuples ``(vertex1, vertex2, type of supplementarity, neighbors)``.
    """
    candidates = g.vertex_set()
    phases = g.phases()

    parities: Dict[FrozenSet[VT],List[VT]] = dict()
    m: List[MatchSupplementarityType[VT]] = []
    taken: Set[VT] = set()
    # First we find all the non-Clifford vertices and their list of neighbors
    while len(candidates) > 0:
        v = candidates.pop()
        if phases[v] == 0 or phases[v].denominator <= 2: continue # Skip Clifford vertices
        neigh = set(g.neighbors(v))
        if not neigh.isdisjoint(taken): continue
        par = frozenset(neigh)
        if par in parities: 
            for w in parities[par]:
                if (phases[v]-phases[w]) % 2 == 1 or (phases[v]+phases[w]) % 2 == 1:
                    m.append((v,w,1,par))
                    taken.update({v,w})
                    taken.update(neigh)
                    candidates.difference_update(neigh)
                    break
            else: parities[par].append(v)
            if v in taken: continue
        else: parities[par] = [v]
        for w in neigh:
            if phases[w] == 0 or phases[w].denominator <= 2 or w in taken: continue
            diff = neigh.symmetric_difference(g.neighbors(w))
            if len(diff) == 2: # Perfect overlap
                if (phases[v] + phases[w]) % 2 == 0 or (phases[v] - phases[w]) % 2 == 1:
                    m.append((v,w,2,frozenset(neigh.difference({w}))))
                    taken.update({v,w})
                    taken.update(neigh)
                    candidates.difference_update(neigh)
                    break
    return m

def apply_supplementarity(
        g: BaseGraph[VT,ET], 
        matches: List[MatchSupplementarityType[VT]]
        ) -> RewriteOutputType[ET,VT]:
    """Given the output of :func:``match_supplementarity``, removes non-Clifford spiders that act on the same set of targets trough supplementarity."""
    rem = []
    for v, w, t, neigh in matches:
        rem.append(v)
        rem.append(w)
        alpha = g.phase(v)
        beta = g.phase(w)
        g.scalar.add_power(-2*len(neigh))
        if t == 1: # v and w are not connected
            g.scalar.add_node(2*alpha+1)
            #if (alpha-beta)%2 == 1: # Standard supplementarity    
            if (alpha+beta)%2 == 1: # Need negation on beta
                g.scalar.add_phase(-alpha + 1)
                for n in neigh:
                    g.add_to_phase(n,1)
        elif t == 2: # they are connected
            g.scalar.add_power(-1)
            g.scalar.add_node(2*alpha)
            #if (alpha-beta)%2 == 1: # Standard supplementarity 
            if (alpha+beta)%2 == 0: # Need negation
                g.scalar.add_phase(-alpha)
                for n in neigh:
                    g.add_to_phase(n,1)
        else: raise Exception("Shouldn't happen")
    return ({}, rem, [], True)


MatchCopyType = Tuple[VT,VT,FractionLike,FractionLike,List[VT]]

def match_copy(
    g: BaseGraph[VT,ET], 
    vertexf:Optional[Callable[[VT],bool]]=None
    ) -> List[MatchCopyType[VT]]:
    """Finds spiders with a 0 or pi phase that have a single neighbor,
    and copies them through. Assumes that all the spiders are green and maximally fused."""
    if vertexf is not None:
        candidates = set([v for v in g.vertices() if vertexf(v)])
    else:
        candidates = g.vertex_set()
    phases = g.phases()
    types = g.types()
    m = []

    while len(candidates) > 0:
        v = candidates.pop()
        if phases[v] not in (0,1) or types[v] != VertexType.Z or g.vertex_degree(v) != 1: continue
        w = list(g.neighbors(v))[0]
        if types[w] != VertexType.Z: continue
        neigh = [n for n in g.neighbors(w) if n != v]
        m.append((v,w,phases[v],phases[w],neigh))
        candidates.discard(w)
        candidates.difference_update(neigh)
    return m

def apply_copy(g: BaseGraph[VT,ET], matches: List[MatchCopyType[VT]]) -> RewriteOutputType[ET,VT]:
    rem = []
    types = g.types()
    outputs = g.outputs()
    for v,w,a,alpha, neigh in matches:
        rem.append(v)
        rem.append(w)
        g.scalar.add_power(-len(neigh)+1)
        if a: g.scalar.add_phase(alpha)
        for n in neigh: 
            if types[n] == VertexType.BOUNDARY:
                r = g.row(n) - 1 if n in outputs else g.row(n)+1
                u = g.add_vertex(VertexType.Z, g.qubit(n), r, a)
                e = g.edge(w,n)
                et = g.edge_type(e)
                g.add_edge(g.edge(n,u), toggle_edge(et))
            g.add_to_phase(n, a)
    return ({}, rem, [], True)


MatchPhasePolyType = Tuple[List[VT], Dict[FrozenSet[VT],Union[VT,Tuple[VT,VT]]]]

def match_gadgets_phasepoly(g: BaseGraph[VT,ET]) -> List[MatchPhasePolyType[VT]]:
    """Finds groups of phase-gadgets that act on the same set of 4 vertices in order to apply a rewrite based on
    rule R_13 of the paper *A Finite Presentation of CNOT-Dihedral Operators*.""" 
    targets: Dict[VT,Set[FrozenSet[VT]]] = {}
    gadgets: Dict[FrozenSet[VT], Tuple[VT,VT]] = {}
    inputs = g.inputs()
    outputs = g.outputs()
    for v in g.vertices():
        if v not in inputs and v not in outputs and len(list(g.neighbors(v)))==1:
            if g.phase(v) != 0 and g.phase(v).denominator != 4: continue
            n = list(g.neighbors(v))[0]
            tgts = frozenset(set(g.neighbors(n)).difference({v}))
            if len(tgts)>4: continue
            gadgets[tgts] = (n,v)
            for t in tgts:
                if t in targets: targets[t].add(tgts)
                else: targets[t] = {tgts}
        if g.phase(v) != 0 and g.phase(v).denominator == 4:
            if v in targets: targets[v].add(frozenset([v]))
            else: targets[v] = {frozenset([v])}
    targets = {t:s for t,s in targets.items() if len(s)>1}
    matches: Dict[FrozenSet[VT], Set[FrozenSet[VT]]] = {}

    for v1,t1 in targets.items():
        s = t1.difference(frozenset([v1]))
        if len(s) == 1:
            c = s.pop()
            if any(len(targets[v2])==2 for v2 in c): continue
        s = t1.difference({frozenset({v1})})
        for c in [d for d in s if not any(d.issuperset(e) for e in s if e!=d)]:
            if not all(v2 in targets for v2 in c): continue
            if any(v2<v1 for v2 in c): continue # type: ignore
            a = set()
            for t in c: a.update([i for s in targets[t] for i in s if i in targets])
            for group in itertools.combinations(a.difference(c),4-len(c)):
                gr = list(group)+list(c)
                b: Set[FrozenSet[VT]] = set()
                for t in gr: b.update([s for s in targets[t] if s.issubset(gr)])
                if len(b)>7:
                    matches[frozenset(gr)] = b

    m: List[MatchPhasePolyType[VT]] = []
    taken: Set[VT] = set()
    for groupp, gad in sorted(matches.items(), key=lambda x: len(x[1]), reverse=True):
        if taken.intersection(groupp): continue
        m.append((list(groupp), {s:(gadgets[s] if len(s)>1 else list(s)[0]) for s in gad}))
        taken.update(groupp)

    return m

def apply_gadget_phasepoly(g: BaseGraph[VT,ET], matches: List[MatchPhasePolyType[VT]]) -> None:
    """Uses the output of :func:`match_gadgets_phasepoly` to apply a rewrite based 
    on rule R_13 of the paper *A Finite Presentation of CNOT-Dihedral Operators*.""" 
    rs = g.rows()
    phases = g.phases()
    for group, gadgets in matches:
        for i in range(4):
            v1 = group[i]
            g.add_to_phase(v1, Fraction(5,4))

            for j in range(i+1,4):
                v2 = group[j]
                f = frozenset({v1,v2})
                if f in gadgets:
                    n,v = gadgets[f] # type: ignore # complex typing situation
                    phase = phases[v]
                    if phases[n]:
                        phase = -phase
                        g.set_phase(n,0)
                else:
                    n = g.add_vertex(VertexType.Z,-1, rs[v2]+0.5)
                    v = g.add_vertex(VertexType.Z,-2, rs[v2]+0.5)
                    phase = 0
                    g.add_edges([g.edge(n,v),g.edge(v1,n),g.edge(v2,n)],EdgeType.HADAMARD)
                g.set_phase(v, phase + Fraction(3,4))

                for k in range(j+1,4):
                    v3 = group[k]
                    f = frozenset({v1,v2,v3})
                    if f in gadgets:
                        n,v = gadgets[f] # type: ignore
                        phase = phases[v]
                        if phases[n]:
                            phase = -phase
                            g.set_phase(n,0)
                    else:
                        n = g.add_vertex(VertexType.Z,-1, rs[v3]+0.5)
                        v = g.add_vertex(VertexType.Z,-2, rs[v3]+0.5)
                        phase = 0
                        g.add_edges([g.edge(*e) for e in [(n,v),(v1,n),(v2,n),(v3,n)]],EdgeType.HADAMARD)
                    g.set_phase(v, phase + Fraction(1,4))
        f = frozenset(group)
        if f in gadgets:
            n,v = gadgets[f] # type: ignore
            phase = phases[v]
            if phases[n]:
                phase = -phase
                g.set_phase(n,0)
        else:
            n = g.add_vertex(1,-1, rs[group[0]]+0.5)
            v = g.add_vertex(1,-2, rs[group[0]]+0.5)
            phase = 0
            g.add_edges([g.edge(n,v)]+[g.edge(n,w) for w in group],EdgeType.HADAMARD)
        g.set_phase(v, phase + Fraction(7,4))


MatchIdFuseType = Tuple[VT,VT,VT]

def match_id_fuse(
        g: BaseGraph[VT,ET], 
        vertexf: Optional[Callable[[VT],bool]] = None, 
        num: int = -1,
        allow_interacting_matches: bool = False
        ) -> List[MatchIdFuseType[VT]]:
    
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    
    i = 0
    m: List[MatchIdFuseType] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        if not vertex_is_zx(g.type(v)) or g.is_ground(v): continue
        
        if g.simplifier:
            if g.check_phase(v,0) is False: continue
        elif g.phase(v) != 0: continue
        
        ns = g.neighbors(v)
        if len(ns) != 2: continue
        v0, v1 = ns
        if not (vertex_is_zx(g.type(v0)) and g.type(v0) == g.type(v1)): continue
        if g.edge_type(g.edge(v,v0)) != g.edge_type(g.edge(v,v1)): continue # Do not put ground spiders on the boundary
        if any(len(g.neighbors(u)) == 1 for u in (v0,v1)): continue # Phase gadget
        
        m.append((v,v0,v1))
        i += 1
        
        # if g.simplifier: g.fix_phase(v,0)
        
        if allow_interacting_matches: continue
        candidates.discard(v0)
        candidates.discard(v1)
        for n in g.neighbors(v0): 
            candidates.discard(n)
            for n2 in g.neighbors(n): candidates.discard(n2)
        for n in g.neighbors(v1): 
            candidates.discard(n)
            for n2 in g.neighbors(n): candidates.discard(n2)
    return m

def id_fuse(g: BaseGraph[VT,ET], matches: List[MatchIdFuseType[VT]]) -> RewriteOutputType[ET,VT]:
    rem_verts = []
    etab: Dict[ET,List[int]] = dict()

    for m in matches:
        rem_verts.append(m[0])
        
        v0,v1 = m[1],m[2]

        if g.is_ground(v0) or g.is_ground(v1):
            g.set_phase(v0, 0)
            g.set_ground(v0)
        else: g.add_to_phase(v0, g.phase(v1))
        
        if g.simplifier:
            g.fix_phase(m[0],0)
            g.fuse_phases(v0,v1)
        
        rem_verts.append(v1) # always delete the second vertex in the match

        for w in g.neighbors(v1): # edges from the second vertex are transferred to the first
            if w == m[0] or w == v0: continue
            e = g.edge(v0,w)
            if e not in etab: etab[e] = [0,0]
            etab[e][g.edge_type(g.edge(v1,w))-1] += 1
            
    return (etab, rem_verts, [], True)


def unfuse_neighbours(g: BaseGraph[VT,ET], v: VT, neighbours_to_unfuse: List[VT], desired_phase: FractionLike) -> Tuple[VT,VT]:
    unfused_phase = split_phases(g.phase(v), desired_phase)
    vp = g.add_vertex(VertexType.Z,-2,g.row(v),unfused_phase)
    v0 = g.add_vertex(VertexType.Z,-1,g.row(v))
    g.set_phase(v,desired_phase)
    g.add_edge((v,v0),EdgeType.HADAMARD)
    g.add_edge((v0,vp),EdgeType.HADAMARD)
    
    for n in neighbours_to_unfuse:
        g.add_edge((vp,n),g.edge_type(g.edge(v,n)))
        g.remove_edge(g.edge(v,n))
        
    if g.simplifier: g.unfuse_vertex(vp, v)
    
    return v0, vp

def split_phases(original_phase: FractionLike, desired_phase: FractionLike) -> FractionLike:
    extend_denom = max(original_phase.denominator,desired_phase.denominator)
    original_phase_n = int(original_phase.numerator*(extend_denom/original_phase.denominator))
    desired_phase_n = int(desired_phase.numerator*(extend_denom/desired_phase.denominator))
    new_phase = Fraction( int((original_phase_n- desired_phase_n) % (extend_denom*2)), extend_denom)
    return new_phase


MatchLcompUnfuseType = Tuple[VT,List[VT],List[VT]]

def match_lcomp_unfuse(
        g: BaseGraph[VT,ET], 
        x,
        heuristic: Callable[[BaseGraph[VT,ET],MatchPivotUnfuseType],int],
        vertexf:Optional[Callable[[VT],bool]] = None, 
        num: int = -1,
        allow_interacting_matches: bool = False
        ) -> Dict[MatchLcompUnfuseType[VT],int]:
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    
    i = 0
    m: Dict[MatchLcompUnfuseType,int]= dict()
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        
        if g.type(v) != VertexType.Z: continue
        if g.is_ground(v): continue
        if g.vertex_degree(v) == 1: continue
        
        vn = list(g.neighbors(v))
        vb = []
        invalid_vertex = False
        for n in vn:
            if g.type(n) == VertexType.Z and g.edge_type(g.edge(v,n)) == EdgeType.HADAMARD: continue
            elif g.type(n) == VertexType.BOUNDARY: vb.append(n)
            else:
                invalid_vertex = True
                break
        if invalid_vertex: continue

        unfusion_matches = get_lcomp_unfusion_matches(g,x,v,vn,vb,heuristic)
        if len(unfusion_matches) == 0: continue
        
        m.update(unfusion_matches)
        i += 1
        
        if allow_interacting_matches: continue
        for n in vn: candidates.discard(n)
    return m

def get_lcomp_unfusion_matches(
        g: BaseGraph[VT,ET],
        x,
        v: VT, 
        vn: List[VT], 
        vb: List[VT], 
        heuristic: Callable[[BaseGraph[VT,ET],MatchPivotUnfuseType],int]
        ) -> Dict[MatchLcompUnfuseType[VT],int]:
    
    unfusion_matches = dict()
    # potential_unfusion_neighbours = [list(subset) for i in range(len(vn)-1) for subset in itertools.combinations(vn, i)]
    potential_unfusion_neighbours = [list(subset) for i in range(2) for subset in itertools.combinations(vn, i)]
    # potential_unfusion_neighbours = []
    
    for neighbours_to_unfuse in potential_unfusion_neighbours:
        if not set(vb) <= set(neighbours_to_unfuse): continue
        
        if len(neighbours_to_unfuse) == 0:
            if g.simplifier:
                if g.check_phase(v,Fraction(1,2)) is False and g.check_phase(v,Fraction(3,2)) is False: continue
            elif g.phase(v) not in (Fraction(1,2),Fraction(3,2)): continue
        
        match = (v,tuple(vn),tuple(neighbours_to_unfuse))
        score = heuristic(g,x,match)
        if score is None: continue
        unfusion_matches[match] = score
    return unfusion_matches

def lcomp_unfuse(g: BaseGraph[VT,ET], matches: List[MatchLcompUnfuseType[VT]]) -> RewriteOutputType[ET,VT]:
    updated_matches = []
    for v, vn, neighbours_to_unfuse in matches:
        if len(neighbours_to_unfuse) == 0:
            if g.simplifier:
                fix_1 = g.check_phase(v, Fraction(1/2))
                if fix_1 is True:
                    fix_3 = g.check_phase(v,Fraction(3,2))
                    if fix_3 is True: g.fix_phase(v, Fraction(1/2))
                    elif fix_3 == 1: g.fix_phase(v, Fraction(3/2))
                elif fix_1 == 1: g.fix_phase(v, Fraction(1/2))
                else: g.fix_phase(v, Fraction(3/2))
            updated_matches.append((v,list(vn)))
            continue
        v0, vp = unfuse_neighbours(g,v,neighbours_to_unfuse,Fraction(1,2))
        updated_vn = [v for v in vn if v not in neighbours_to_unfuse] + [v0]
        updated_matches.append((v,updated_vn))
    return lcomp(g, updated_matches)


MatchPivotUnfuseType = Tuple[VT,VT,List[VT],List[VT],Tuple[VT,VT]]

def match_pivot_unfuse(
        g: BaseGraph[VT,ET], 
        x,
        heuristic: Callable[[BaseGraph[VT,ET],MatchPivotUnfuseType],int],
        matchf: Optional[Callable[[ET],bool]] = None, 
        num: int = -1,
        check_edge_types: bool = True,
        allow_interacting_matches: bool = False
        ) -> Dict[MatchPivotUnfuseType[VT],int]:
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    
    i = 0
    m: Dict[MatchPivotUnfuseType,int] = dict()
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if check_edge_types and g.edge_type(e) != EdgeType.HADAMARD: continue
        
        v0, v1 = g.edge_st(e)
        if not (g.type(v0) == VertexType.Z and g.type(v1) == VertexType.Z): continue
        if g.is_ground(v0) or g.is_ground(v1): continue
        if any(len(g.neighbors(v)) == 1 for v in (v0,v1)): continue
        if any(len(g.neighbors(n)) == 1 for v in (v0,v1) for n in g.neighbors(v)): continue
        
        invalid_edge = False
        v0n = g.neighbors(v0)
        v0b = []
        for n in v0n:
            if g.type(n) == VertexType.Z and any(len(g.neighbors(n2))==1 for n2 in g.neighbors(n)): v0b.append(n)
            elif g.type(n) == VertexType.Z and g.edge_type(g.edge(v0,n)) == EdgeType.HADAMARD: continue
            elif g.type(n) == VertexType.BOUNDARY: v0b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge: continue
        
        v1n = g.neighbors(v1)
        v1b = []
        for n in v1n:
            if g.type(n) == VertexType.Z and any(len(g.neighbors(n2))==1 for n2 in g.neighbors(n)): v1b.append(n)
            elif g.type(n) == VertexType.Z and g.edge_type(g.edge(v1,n)) == EdgeType.HADAMARD: continue
            elif g.type(n) == VertexType.BOUNDARY: v1b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge: continue
        
        unfusion_matches = get_pivot_unfusion_matches(g,x,v0,v1,v0n,v1n,v0b,v1b,heuristic)
        if len(unfusion_matches) == 0: continue

        m.update(unfusion_matches)
        i += 1

        if allow_interacting_matches: continue
        for v in v0n:
            for c in g.incident_edges(v): candidates.discard(c)
        for v in v1n:
            for c in g.incident_edges(v): candidates.discard(c)
    return m

def get_pivot_unfusion_matches(g: BaseGraph[VT,ET], x, v0: VT, v1: VT, v0n, v1n, v0b: List[VT], v1b: List[VT], heuristic):
    unfusion_matches = dict()
    # potential_unfusion_neighbors_v0 = [list(subset) for i in range(len(v0n)-1) for subset in itertools.combinations(v0n, i)]
    # potential_unfusion_neighbors_v1 = [list(subset) for i in range(len(v1n)-1) for subset in itertools.combinations(v1n, i)]
    potential_unfusion_neighbors_v0 = [list(subset) for i in range(2) for subset in itertools.combinations(v0n, i)]
    potential_unfusion_neighbors_v1 = [list(subset) for i in range(2) for subset in itertools.combinations(v1n, i)]
    potential_unfusion_neighbors = [(n0,n1) for n0 in potential_unfusion_neighbors_v0 for n1 in potential_unfusion_neighbors_v1]
    # potential_unfusion_neighbors = [([],[])]
    
    for neighbours_to_unfuse in potential_unfusion_neighbors:
        if v0 in neighbours_to_unfuse[1] or v1 in neighbours_to_unfuse[0]: continue
        if not set(v0b) <= set(neighbours_to_unfuse[0]): continue
        if not set(v1b) <= set(neighbours_to_unfuse[1]): continue
        if len(neighbours_to_unfuse[0]) == 0 and len(neighbours_to_unfuse[1]) == 0:
            if g.simplifier:
                if any(p is False for p in g.check_two_pauli_phases(v0, v1)): continue
            elif any(g.phase(v) not in (0,1) for v in (v0,v1)): continue
        elif len(neighbours_to_unfuse[0]) == 0:
            if g.simplifier:
                if g.check_phase(v0,0) is False and g.check_phase(v0,1) is False: continue
            elif g.phase(v0) not in (0,1): continue
        elif len(neighbours_to_unfuse[1]) == 0:
            if g.simplifier:
                if g.check_phase(v1,0) is False and g.check_phase(v1,1) is False: continue
            elif g.phase(v1) not in (0,1): continue
            
        match = (v0,v1,tuple(v0b),tuple(v1b),tuple(tuple(ns) for ns in neighbours_to_unfuse))
        score = heuristic(g, x, match)
        if score is None: continue
        unfusion_matches[match] = score
    
    return unfusion_matches

def pivot_unfuse(g: BaseGraph[VT,ET], matches: List[MatchPivotUnfuseType[VT]]) -> RewriteOutputType[ET,VT]:
    updated_matches = []
    for v0, v1, v0b, v1b, neighbours_to_unfuse in matches:
        if len(neighbours_to_unfuse[0]) == 0 and len(neighbours_to_unfuse[1]) == 0:
            if g.simplifier:
                phases = g.check_two_pauli_phases(v0,v1)
                g.fix_phase(v0,phases[0])
                g.fix_phase(v1,phases[1])
            updated_matches.append((v0,v1,v0b,v1b))
            continue
        if len(neighbours_to_unfuse[0]) == 0:
            if g.simplifier:
                fix_0 = g.check_phase(v0, 0)
                if fix_0 is True:
                    fix_1 = g.check_phase(v0,1)
                    if fix_1 is True: g.fix_phase(v0,0)
                    elif fix_1 == 1: g.fix_phase(v0,1)
                elif fix_0 == 1: g.fix_phase(v0,0)
                else: g.fix_phase(v0,1)
            unfuse_neighbours(g,v1,neighbours_to_unfuse[1],0)
            v1b = list(set(v1b).difference(set(neighbours_to_unfuse[1])))
            updated_matches.append((v0,v1,v0b,v1b))
            continue
        if len(neighbours_to_unfuse[1]) == 0:
            if g.simplifier:
                fix_0 = g.check_phase(v1, 0)
                if fix_0 is True:
                    fix_1 = g.check_phase(v1,1)
                    if fix_1 is True: g.fix_phase(v1,0)
                    elif fix_1 == 1: g.fix_phase(v1,1)
                elif fix_0 == 1: g.fix_phase(v1,0)
                else: g.fix_phase(v1,1)
            unfuse_neighbours(g,v0,neighbours_to_unfuse[0],0)
            v0b = list(set(v0b).difference(set(neighbours_to_unfuse[0])))
            updated_matches.append((v0,v1,v0b,v1b))
            continue
        unfuse_neighbours(g,v0,neighbours_to_unfuse[0],0)
        v0b = list(set(v0b).difference(set(neighbours_to_unfuse[0])))
        unfuse_neighbours(g,v1,neighbours_to_unfuse[1],0)
        v1b = list(set(v1b).difference(set(neighbours_to_unfuse[1])))
        updated_matches.append((v0,v1,v0b,v1b))
    return pivot(g, updated_matches)


MatchUnfuseType = Tuple[Optional[MatchLcompUnfuseType],Optional[MatchPivotUnfuseType]]

def match_2Q_reduce(g: BaseGraph[VT,ET], x, matchf: Optional[Callable[[VT],bool]] = None) -> Dict[MatchUnfuseType[VT],int]:
    m = dict()
    m.update({(match,None,None):v for match,v in match_lcomp_unfuse(g,x,lcomp_2Q_reduce_heuristic,matchf,allow_interacting_matches=True).items()})
    m.update({(None,match,None):v for match,v in match_pivot_unfuse(g,x,pivot_2Q_reduce_heuristic,matchf,allow_interacting_matches=True).items()})
    m.update({(None,None,match):id_fuse_2Q_reduce_heuristic(g,x,match) for match in match_id_fuse(g,matchf,allow_interacting_matches=True)})
    return m

def remove_updated_2Q_reduce_matches(current_matches: Dict[MatchUnfuseType[VT],int],removed_vertices: List[VT], verts_to_update: List[VT]):
    updated_matches = dict()
    for m, val in current_matches.items():
        if m[0]:
            if m[0][0] in verts_to_update or m[0][0] in removed_vertices: continue
        elif m[1]:
            if all(v in verts_to_update for v in [m[1][0],m[1][1]]) or any(v in removed_vertices for v in [m[1][0],m[1][1]]): continue
        elif m[2]:
            if m[2][0] in verts_to_update or m[2][0] in removed_vertices: continue
        else: continue
        updated_matches[m] = val
    return updated_matches

def unfuse(g: BaseGraph[VT,ET], match: MatchUnfuseType) -> RewriteOutputType[ET,VT]:
    if match[0]: return lcomp_unfuse(g,[match[0]])
    if match[1]: return pivot_unfuse(g,[match[1]])
    if match[2]: return id_fuse(g,[match[2]])

def match_int_cliff(g, x=None, matchf=None):
    m = dict()
    for match in match_lcomp_parallel(g,vertexf=matchf,allow_interacting_matches=True):
        edges_removed, vertices_removed = lcomp_statistics(g,match[0],match[1],[])
        twoQ_removed = edges_removed - vertices_removed
        if twoQ_removed < 0: continue
        m[(match,None,None)] = x[0]*(twoQ_removed + x[3]*vertices_removed + x[4])
    for match in match_pivot_parallel(g,matchf=matchf,allow_interacting_matches=True):
        edges_removed, vertices_removed = pivot_statistics(g,match[0],match[1],match[2],match[3],[[],[]])
        twoQ_removed = edges_removed - vertices_removed
        if twoQ_removed < 0: continue
        m[(None,match,None)] = x[1]*(twoQ_removed + x[5]*vertices_removed + x[6])
    for match in match_id_fuse(g,matchf,allow_interacting_matches=True):
        edges_removed, vertices_removed = id_fuse_statistics(g,match[0],match[1],match[2])
        if twoQ_removed < 0: continue
        x[2]*(twoQ_removed + x[7]*vertices_removed + x[8])
    return m

def int_cliff(g,match):
    if match[0]: return lcomp(g,[match[0]])
    if match[1]: return pivot(g,[match[1]])
    if match[2]: return id_fuse(g,[match[2]])