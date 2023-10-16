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

"""This module contains the ZX-diagram simplification strategies of PyZX.
Each strategy is based on applying some combination of the rewrite rules in the rules_ module.
The main procedures of interest are :func:`clifford_simp` for simple reductions,
:func:`full_reduce` for the full rewriting power of PyZX, and :func:`teleport_reduce` to
use the power of :func:`full_reduce` while not changing the structure of the graph.
"""

__all__ = ['bialg_simp','spider_simp', 'id_simp', 'phase_free_simp', 'pivot_simp',
        'pivot_gadget_simp', 'pivot_boundary_simp', 'gadget_simp',
        'lcomp_simp', 'clifford_simp', 'tcount', 'to_gh', 'to_rg',
        'full_reduce', 'teleport_reduce', 'reduce_scalar', 'supplementarity_simp']

from typing import List, Callable, Optional, Union, Generic, Tuple, Dict, Iterator, Any

from .utils import EdgeType, VertexType, toggle_edge, vertex_is_zx, toggle_vertex
from .rules import *
from .flow import *
from .graph.base import BaseGraph, VT, ET
from .circuit import Circuit
from .drawing import draw

class Stats(object):
    def __init__(self) -> None:
        self.num_rewrites: Dict[str,int] = {}
    def count_rewrites(self, rule: str, n: int) -> None:
        if rule in self.num_rewrites:
            self.num_rewrites[rule] += n
        else:
            self.num_rewrites[rule] = n
    def __str__(self) -> str:
        s = "REWRITES\n"
        nt = 0
        for r,n in self.num_rewrites.items():
            nt += n
            s += "%s %s\n" % (str(n).rjust(6),r)
        s += "%s TOTAL" % str(nt).rjust(6)
        return s

def simp(
    g: BaseGraph[VT,ET],
    name: str,
    match: Callable[..., List[MatchObject]],
    rewrite: Callable[[BaseGraph[VT,ET],List[MatchObject]], RewriteOutputType[ET,VT]],
    matchf: Union[Optional[Callable[[VT],bool]],Optional[Callable[[ET],bool]]]=None,
    max_num_rewrites: Optional[int] = None,
    quiet:bool=False,
    stats:Optional[Stats]=None) -> int:
    """Helper method for constructing simplification strategies based on the rules present in rules_.
    It uses the ``match`` function to find matches, and then rewrites ``g`` using ``rewrite``.
    If ``matchf`` is supplied, only the vertices or edges for which matchf() returns True are considered for matches.

    Example:
        ``simp(g, 'spider_simp', rules.match_spider_parallel, rules.spider)``

    Args:
        g: The graph that needs to be simplified.
        str name: The name to display if ``quiet`` is set to False.
        match: One of the ``match_*`` functions of rules_.
        rewrite: One of the rewrite functions of rules_.
        matchf: An optional filtering function on candidate vertices or edges, which
           is passed as the second argument to the match function.
        condition: An optional filtering function on the graph produced by individual rewrite rules (e.g. edges created < edges removed)
        quiet: Suppress output on numbers of matches found during simplification.

    Returns:
        Number of iterations of ``rewrite`` that had to be applied before no more matches were found."""

    num_iterations = 0
    num_rewrites = 0
    total_rewrites = 0
    while True:
        if matchf and max_num_rewrites: matches = match(g, matchf, num = -1 if max_num_rewrites == -1 else max_num_rewrites-total_rewrites)
        elif matchf: matches = match(g, matchf)
        elif max_num_rewrites: matches = match(g, num = -1 if max_num_rewrites == -1 else max_num_rewrites-total_rewrites)
        else: matches = match(g)
        
        num_rewrites = len(matches)
        if num_rewrites == 0: break
        
        num_iterations += 1
        total_rewrites += num_rewrites
        if num_iterations == 1 and not quiet: print(f'{name}: ',end='')
        if not quiet: print(num_rewrites, end='')
        apply_rule(g,rewrite,matches)
        if not quiet: print('. ', end='')
        if stats is not None: stats.count_rewrites(name, num_rewrites)
        if total_rewrites == max_num_rewrites: break
        
    if not quiet and num_iterations>0: print(f' {num_iterations} iterations')
    return num_iterations

def pivot_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[ET],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'pivot_simp', match_pivot_parallel, pivot, matchf=matchf, quiet=quiet, stats=stats)

def pivot_gadget_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[ET],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'pivot_gadget_simp', match_pivot_gadget, pivot_gadget, matchf=matchf, quiet=quiet, stats=stats)

def pivot_boundary_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[ET],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'pivot_boundary_simp', match_pivot_boundary, pivot_gadget, matchf=matchf, quiet=quiet, stats=stats)

def lcomp_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[VT],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'lcomp_simp', match_lcomp_parallel, lcomp, matchf=matchf, quiet=quiet, stats=stats)

def bialg_simp(g: BaseGraph[VT,ET], quiet:bool=False, stats: Optional[Stats]=None) -> int:
    return simp(g, 'bialg_simp', match_bialg_parallel, bialg, quiet=quiet, stats=stats)

def spider_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[VT],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'spider_simp', match_spider_parallel, spider, matchf=matchf, quiet=quiet, stats=stats)

def id_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[VT],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'id_simp', match_ids_parallel, remove_ids, matchf=matchf, quiet=quiet, stats=stats)

def id_fuse_simp(g: BaseGraph[VT,ET], matchf:Optional[Callable[[VT],bool]]=None, quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'id_fuse', match_id_fuse, id_fuse, matchf=matchf, quiet=quiet, stats=stats)

def gadget_simp(g: BaseGraph[VT,ET], quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'gadget_simp', match_phase_gadgets, merge_phase_gadgets, quiet=quiet, stats=stats)

def supplementarity_simp(g: BaseGraph[VT,ET], quiet:bool=False, stats:Optional[Stats]=None) -> int:
    return simp(g, 'supplementarity_simp', match_supplementarity, apply_supplementarity, quiet=quiet, stats=stats)

def copy_simp(g: BaseGraph[VT,ET], quiet:bool=False, stats:Optional[Stats]=None) -> int:
    """Copies 1-ary spiders with 0/pi phase through neighbors.
    WARNING: only use on maximally fused diagrams consisting solely of Z-spiders."""
    return simp(g, 'copy_simp', match_copy, apply_copy, quiet=quiet, stats=stats)

def phase_free_simp(g:BaseGraph[VT,ET], quiet:bool=False, stats:Optional[Stats]=None) -> int:
    '''Performs the following set of simplifications on the graph:
    spider -> bialg'''
    i1 = spider_simp(g, quiet=quiet, stats=stats)
    i2 = bialg_simp(g, quiet=quiet, stats=stats)
    return i1+i2

def basic_simp(g:BaseGraph[VT,ET], quiet: bool=True, stats:Optional[Stats] = None) -> int:
    """Keeps doing the simplifications ``id_simp``, ``spider_simp`` until none of them can be applied anymore."""
    j = 0
    while True:
        i1 = id_simp(g, quiet=quiet, stats=stats)
        i2 = spider_simp(g, quiet=quiet, stats=stats)
        if i1 + i2 == 0: break
        j += 1
    return j

def interior_clifford_simp(g: BaseGraph[VT,ET], quiet:bool=False, stats:Optional[Stats]=None) -> int:
    """Keeps doing the simplifications ``id_simp``, ``spider_simp``,
    ``pivot_simp`` and ``lcomp_simp`` until none of them can be applied anymore."""
    spider_simp(g, quiet=quiet, stats=stats)
    to_gh(g)
    i = 0
    while True:
        i1 = id_simp(g, quiet=quiet, stats=stats)
        i2 = spider_simp(g, quiet=quiet, stats=stats)
        i3 = pivot_simp(g, quiet=quiet, stats=stats)
        i4 = lcomp_simp(g, quiet=quiet, stats=stats)
        if i1+i2+i3+i4==0: break
        i += 1
    return i

def clifford_simp(g: BaseGraph[VT,ET], quiet:bool=True, stats:Optional[Stats]=None) -> int:
    """Keeps doing rounds of :func:`interior_clifford_simp` and
    :func:`pivot_boundary_simp` until they can't be applied anymore."""
    i = 0
    while True:
        i += interior_clifford_simp(g, quiet=quiet, stats=stats)
        i2 = pivot_boundary_simp(g, quiet=quiet, stats=stats)
        if i2 == 0:
            break
    return i

def reduce_scalar(g: BaseGraph[VT,ET], quiet:bool=True, stats:Optional[Stats]=None) -> int:
    """Modification of ``full_reduce`` that is tailered for scalar ZX-diagrams.
    It skips the boundary pivots, and it additionally does ``supplementarity_simp``."""
    i = 0
    while True:
        i1 = id_simp(g, quiet=quiet, stats=stats)
        i2 = spider_simp(g, quiet=quiet, stats=stats)
        i3 = pivot_simp(g, quiet=quiet, stats=stats)
        i4 = lcomp_simp(g, quiet=quiet, stats=stats)
        if i1+i2+i3+i4:
            i += 1
            continue
        i5 = pivot_gadget_simp(g,quiet=quiet, stats=stats)
        i6 = gadget_simp(g, quiet=quiet, stats=stats)
        if i5 + i6:
            i += 1
            continue
        i7 = supplementarity_simp(g,quiet=quiet, stats=stats)
        if not i7: break
        i += 1
    return i

def full_reduce(g: BaseGraph[VT,ET], quiet:bool=True, stats:Optional[Stats]=None) -> None:
    """The main simplification routine of PyZX. It uses a combination of :func:`clifford_simp` and
    the gadgetization strategies :func:`pivot_gadget_simp` and :func:`gadget_simp`."""
    interior_clifford_simp(g, quiet=quiet, stats=stats)
    pivot_gadget_simp(g,quiet=quiet, stats=stats)
    while True:
        clifford_simp(g,quiet=quiet, stats=stats)
        i = gadget_simp(g, quiet=quiet, stats=stats)
        interior_clifford_simp(g,quiet=quiet, stats=stats)
        j = pivot_gadget_simp(g,quiet=quiet, stats=stats)
        if i+j == 0: 
            break

def teleport_reduce(g: BaseGraph[VT,ET], store: bool = False) -> None:
    """This simplification procedure runs :func:`full_reduce` in a way
    that does not change the graph structure of the resulting diagram.
    The only thing that is different in the output graph are the location and value of the phases."""
    s = PhaseTeleporter(g)
    s.teleport_phases(store = store)
    g.replace(s.simplify_graph)

class PhaseTeleporter(Generic[VT, ET]):
    """Class used for :func:`teleport_reduce`."""
    def __init__(self, g: BaseGraph[VT,ET]) -> None:
        self.original_graph: BaseGraph[VT,ET] = g.copy()
        self.parent_vertex: Dict[VT,VT] = {}
        self.vertex_rank: Dict[VT,int] = {}
        self.phase_mult: Dict[VT,int] = {}
        self.non_clifford_vertices: Set[VT] = set()
        for v in self.original_graph.vertices():
            if self.original_graph.phase(v).denominator > 2:
                self.parent_vertex[v] = v
                self.vertex_rank[v] = 0
                self.phase_mult[v] = 1
                self.non_clifford_vertices.add(v)
    
    def parent(self, v: VT) -> VT:
        if self.parent_vertex[v] != v:
            self.parent_vertex[v] = self.parent(self.parent_vertex[v])
        return self.parent_vertex[v]
    
    def get_vertex_groups(self) -> List[List[VT]]:
        vertex_groups: Dict[VT,List[VT]] = {}
        for v in self.non_clifford_vertices:
            root = self.parent(v)
            if root not in vertex_groups:
                vertex_groups[root] = []
            vertex_groups[root].append(v)
        return list(vertex_groups.values())
    
    def fuse_phases(self, v1: VT, v2: VT) -> None:
        if not all(v in self.non_clifford_vertices for v in (v1, v2)): return
        root_v1 = self.parent(v1)
        root_v2 = self.parent(v2)
        if root_v1 != root_v2:
            if self.vertex_rank[root_v1] > self.vertex_rank[root_v2]:
                self.parent_vertex[root_v2] = root_v1
            else:
                self.parent_vertex[root_v1] = root_v2
                if self.vertex_rank[root_v1] == self.vertex_rank[root_v2]:
                    self.vertex_rank[root_v2] += 1
    
    def phase_negate(self, v: VT) -> None:
        root = self.parent(v)
        for vert in self.non_clifford_vertices:
            if self.parent(vert) == root:
                self.phase_mult[vert] *= -1
    
    def init_simplify_graph(self, fusing_mode: bool = True) -> None:
        self.simplify_graph = self.original_graph.clone()
        self.simplify_graph.set_phase_teleporter(self, fusing_mode)
    
    def teleport_phases(self, store:bool = False) -> None:
        self.init_simplify_graph()
        full_reduce(self.simplify_graph)
        self.init_simplify_graph(fusing_mode = False)
        if not store: self.simplify_graph.place_tracked_phases()

def selective_simp(
    g: BaseGraph[VT,ET],
    get_matches: Callable[..., Dict[MatchObject,float]],
    update_matches: Callable[..., Dict[MatchObject,float]],
    rewrite: Callable[[BaseGraph[VT,ET],List[MatchObject]],RewriteOutputType[ET,VT]],
    matchf: Union[Optional[Callable[[VT],bool]],Optional[Callable[[ET],bool]]]=None,
    condition: Callable[...,bool] = lambda *a, **kw: True,
    max_num_rewrites:int = -1,
    **kwargs: Any) -> int:
    
    matches = get_matches(g, matchf, **kwargs)
    unchecked_matches = matches.copy()
    num_rewrites = 0
    while True:
        if len(unchecked_matches) == 0: break
        m = max(unchecked_matches, key=unchecked_matches.__getitem__)
        check_g = g.clone()
        apply_rule(check_g, rewrite, [m])
        if condition(check_g, m):
            num_rewrites += 1
            updated_matches = update_matches(g, check_g, matches, get_matches, matchf, **kwargs)
            g.replace(check_g)
            matches = updated_matches.copy()
            unchecked_matches = matches.copy()
            if num_rewrites == max_num_rewrites: break
            continue
        del unchecked_matches[m]
    return num_rewrites

def flow_2Q_simp(
    g: BaseGraph[VT,ET],
    matchf: Union[Optional[Callable[[VT],bool]],Optional[Callable[[ET],bool]]] = None,
    cFlow: bool = True,
    rewrites: List[str] = ['id_fuse','lcomp','pivot'],
    score_params: List[float] = [1,1,1],
    max_lc_unfusions: int = 0,
    max_p_unfusions: int = 0) -> int:
    assert(is_graph_like(g))
    g.vertices_to_update = []
    if cFlow: flow_condition = lambda graph, match: True if match[2] else cflow(graph) is not None
    else: flow_condition = lambda graph, match: gflow(graph) is not None if match[0] and len(match[0][2])!=0 else gflow(graph) is not None if match[1] and (len(match[1][4][0]) != 0 or len(match[1][4][1]) != 0) else True
    return selective_simp(g, match_2Q_simp, update_2Q_simp_matches, rewrite_2Q_simp, matchf, flow_condition, rewrites=rewrites, score_params = score_params, max_lc_unfusions = max_lc_unfusions, max_p_unfusions = max_p_unfusions) #type:ignore

def update_2Q_simp_matches(
    g_before: BaseGraph[VT,ET],
    g_after: BaseGraph[VT,ET],
    current_matches: Dict[MatchUnfuseType,float],
    get_matches: Callable[..., Dict[MatchUnfuseType,float]],
    matchf: Union[Optional[Callable[[VT],bool]],Optional[Callable[[ET],bool]]] = None,
    **kwargs: Any) -> Dict[MatchUnfuseType,float]:
    
    verts_to_update = set()
    edges_to_update = set()
    
    changed_vertices = g_after.vertices_to_update
    removed_vertices = set(g_before.vertices()).difference(set(g_after.vertices()))
    for v in g_after.vertices():
        if set([e for e in g_before.edges() if v in e]) != set([e for e in g_after.edges() if v in e]) or v in changed_vertices:
            verts_to_update.add(v)
            verts_to_update.update(list(g_after.neighbors(v)))
    
    for v1 in verts_to_update:
        for v2 in verts_to_update:
            if v1 == v2: continue
            if g_after.connected(v1,v2): edges_to_update.add(g_after.edge(v1,v2))
    
    matches_to_update = verts_to_update.union(edges_to_update)
    if matchf: update_m = lambda y: y in matches_to_update and matchf(y)
    else: update_m = lambda y: y in matches_to_update
    new_matches = get_matches(g_after, update_m, **kwargs)
    
    updated_matches = {}
    for m, val in current_matches.items():
        if m[0]:
            if m[0][0] in verts_to_update or m[0][0] in removed_vertices: continue
        elif m[1]:
            if all(v in verts_to_update for v in [m[1][0],m[1][1]]) or any(v in removed_vertices for v in [m[1][0],m[1][1]]): continue
        elif m[2]:
            if m[2][0] in verts_to_update or m[2][0] in removed_vertices: continue
        else: continue
        updated_matches[m] = val
    
    updated_matches.update(new_matches)
    g_after.vertices_to_update = []
    return updated_matches

def to_gh(g: BaseGraph[VT,ET],quiet:bool=True) -> None:
    """Turns every red node into a green node by changing regular edges into hadamard edges"""
    ty = g.types()
    for v in g.vertices():
        if ty[v] == VertexType.X:
            g.set_type(v, VertexType.Z)
            for e in g.incident_edges(v):
                et = g.edge_type(e)
                g.set_edge_type(e, toggle_edge(et))

def to_rg(g: BaseGraph[VT,ET], select:Optional[Callable[[VT],bool]]=None) -> None:
    """Turn green nodes into red nodes by color-changing vertices which satisfy the predicate ``select``.
    By default, the predicate is set to greedily reducing the number of Hadamard-edges.
    :param g: A ZX-graph.
    :param select: A function taking in vertices and returning ``True`` or ``False``."""
    if select is None:
        select = lambda v: (
            len([e for e in g.incident_edges(v) if g.edge_type(e) == EdgeType.SIMPLE]) <
            len([e for e in g.incident_edges(v) if g.edge_type(e) == EdgeType.HADAMARD])
            )

    ty = g.types()
    for v in g.vertices():
        if select(v) and vertex_is_zx(ty[v]):
            g.set_type(v, toggle_vertex(ty[v]))
            for e in g.incident_edges(v):
                g.set_edge_type(e, toggle_edge(g.edge_type(e)))

def is_graph_like(g):
    """Checks if a ZX-diagram is graph-like"""

    # checks that all spiders are Z-spiders
    for v in g.vertices():
        if g.type(v) not in [VertexType.Z, VertexType.BOUNDARY]:
            return False,1

    for v1, v2 in itertools.combinations(g.vertices(), 2):
        if not g.connected(v1, v2):
            continue

        # Z-spiders are only connected via Hadamard edges
        if g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z \
           and g.edge_type(g.edge(v1, v2)) != EdgeType.HADAMARD:
            return False,2

        # FIXME: no parallel edges

    # # no self-loops
    # for v in g.vertices():
    #     if g.connected(v, v):
    #         return False,3

    # # every I/O is connected to a Z-spider
    # bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    # for b in bs:
    #     if g.vertex_degree(b) != 1 or g.type(list(g.neighbors(b))[0]) != VertexType.Z:
    #         return False,4

    # # every Z-spider is connected to at most one I/O
    # zs = [v for v in g.vertices() if g.type(v) == VertexType.Z]
    # for z in zs:
    #     b_neighbors = [n for n in g.neighbors(z) if g.type(n) == VertexType.BOUNDARY]
    #     if len(b_neighbors) > 1:
    #         return False,5,z

    return True


def to_graph_like(g):
    """Puts a ZX-diagram in graph-like form"""

    # turn all red spiders into green spiders
    to_gh(g)

    # simplify: remove excess HAD's, fuse along non-HAD edges, remove parallel edges and self-loops
    spider_simp(g, quiet=True)
    # # ensure all I/O are connected to a Z-spider
    # bs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    # for v in bs:

    #     # if it's already connected to a Z-spider, continue on
    #     if any([g.type(n) == VertexType.Z for n in g.neighbors(v)]):
    #         continue

    #     # have to connect the (boundary) vertex to a Z-spider
    #     ns = list(g.neighbors(v))
    #     for n in ns:
    #         # every neighbor is another boundary or an H-Box
    #         assert(g.type(n) in [VertexType.BOUNDARY, VertexType.H_BOX])
    #         if g.type(n) == VertexType.BOUNDARY:
    #             z1 = g.add_vertex(ty=VertexType.Z)
    #             z2 = g.add_vertex(ty=VertexType.Z)
    #             z3 = g.add_vertex(ty=VertexType.Z)
    #             g.remove_edge(g.edge(v, n))
    #             g.add_edge(g.edge(v, z1), edgetype=EdgeType.SIMPLE)
    #             g.add_edge(g.edge(z1, z2), edgetype=EdgeType.HADAMARD)
    #             g.add_edge(g.edge(z2, z3), edgetype=EdgeType.HADAMARD)
    #             g.add_edge(g.edge(z3, n), edgetype=EdgeType.SIMPLE)
    #         else: # g.type(n) == VertexType.H_BOX
    #             z = g.add_vertex(ty=VertexType.Z)
    #             g.remove_edge(g.edge(v, n))
    #             g.add_edge(g.edge(v, z), edgetype=EdgeType.SIMPLE)
    #             g.add_edge(g.edge(z, n), edgetype=EdgeType.SIMPLE)

    # # each Z-spider can only be connected to at most 1 I/O
    # vs = list(g.vertices())
    # for v in vs:
    #     if not g.type(v) == VertexType.Z:
    #         continue
    #     boundary_ns = [n for n in g.neighbors(v) if g.type(n) == VertexType.BOUNDARY]
    #     if len(boundary_ns) <= 1:
    #         continue

    #     # add dummy spiders for all but one
    #     for b in boundary_ns[:-1]:
    #         z1 = g.add_vertex(ty=VertexType.Z)
    #         z2 = g.add_vertex(ty=VertexType.Z)

    #         g.remove_edge(g.edge(v, b))
    #         g.add_edge(g.edge(z1, z2), edgetype=EdgeType.HADAMARD)
    #         g.add_edge(g.edge(b, z1), edgetype=EdgeType.SIMPLE)
    #         g.add_edge(g.edge(z2, v), edgetype=EdgeType.HADAMARD)

    assert(is_graph_like(g))

def tcount(g: Union[BaseGraph[VT,ET], Circuit]) -> int:
    """Returns the amount of nodes in g that have a non-Clifford phase."""
    if isinstance(g, Circuit):
        return g.tcount()
    count = 0
    phases = g.phases()
    for v in g.vertices():
        if phases[v]!=0 and phases[v].denominator > 2:
            count += 1
    return count

#The functions below haven't been updated in a while. Use at your own risk.

def simp_iter(
        g: BaseGraph[VT,ET],
        name: str,
        match: Callable[..., List[MatchObject]],
        rewrite: Callable[[BaseGraph[VT,ET],List[MatchObject]],RewriteOutputType[ET,VT]]
        ) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    """Version of :func:`simp` that instead of performing all rewrites at once, returns an iterator."""
    i = 0
    new_matches = True
    while new_matches:
        i += 1
        new_matches = False
        m = match(g)
        if len(m) > 0:
            etab, rem_verts, rem_edges, check_isolated_vertices = rewrite(g, m)
            g.add_edge_table(etab)
            g.remove_edges(rem_edges)
            g.remove_vertices(rem_verts)
            if check_isolated_vertices: g.remove_isolated_vertices()
            yield g, name+str(i)
            new_matches = True

def pivot_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    return simp_iter(g, 'pivot', match_pivot_parallel, pivot)

def lcomp_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    return simp_iter(g, 'lcomp', match_lcomp_parallel, lcomp)

def bialg_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    return simp_iter(g, 'bialg', match_bialg_parallel, bialg)

def spider_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    return simp_iter(g, 'spider', match_spider_parallel, spider)

def id_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    return simp_iter(g, 'id', match_ids_parallel, remove_ids)

def clifford_iter(g: BaseGraph[VT,ET]) -> Iterator[Tuple[BaseGraph[VT,ET],str]]:
    yield from spider_iter(g)
    to_gh(g)
    yield g, "to_gh"
    yield from spider_iter(g)
    yield from pivot_iter(g)
    yield from lcomp_iter(g)
    yield from pivot_iter(g)
    yield from id_iter(g)
    yield from spider_iter(g)