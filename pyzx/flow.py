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

from typing import Dict, Set, Tuple, Optional, List, FrozenSet
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
        if g.type(inp) == VertexType.BOUNDARY: pattern_inputs |= set(g.neighbors(inp))
        else: pattern_inputs.add(inp)
    k: int = 1

    for v in processed: l[v] = 0

    while True:
        correct = set()
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
            if not candidates: return True
            return False
        else:
            processed.update(correct)
            k += 1
            
def causal_flow(
    g: BaseGraph[VT, ET],
    update_graph: bool = False
):
    """Compute the causal flow of a diagram in graph-like form.

    Based on an algorithm by Niel de Beaudrap, extended to include phase gadgets.
    See https://doi.org/10.48550/arXiv.quant-ph/0603072
    """
    gadgets, gadget_connections = find_gadgets(g)
    if len(gadgets) == 0:
      flow = fast_flow(g)
      if flow:
        successor_function = flow[1]
        if update_graph: g.update_flow(successor_function)
        return True
      return False
    
    g_without_gadgets = remove_gadgets(g,gadgets)
    io_vertices = []
    for v in g_without_gadgets.inputs():
      if v in g_without_gadgets.outputs():
        io_vertices.append(v)
        g_without_gadgets.remove_vertex(v)
    if len(g_without_gadgets.vertices()) == 0:
      return dict()
    
    k = g_without_gadgets.num_inputs()
    n = g_without_gadgets.num_vertices()
    if g_without_gadgets.num_edges() > (k*n - comb(k+1,2)): return False
    path_cover = BuildPathCover(g_without_gadgets)
    if not path_cover: return False
    successor_function, P, L = GetChainDecomp(g_without_gadgets,path_cover)
    sup = ComputeSuprema(g_without_gadgets,successor_function,P,L)
    if not sup: return False
    
    for n in gadgets.keys():
      connecting = gadget_connections[n]
      for m in gadgets.keys():
        connecting_m = gadget_connections[m]
        first = None
        for v_n in connecting:
          for v_m in connecting_m:
            if v_n == v_m: continue
            if v_n not in P.keys() or v_m not in P.keys(): return False # gadgets are connected
            if n == m and P[v_n] == P[v_m]: return False
            if v_n in g.inputs() or v_m in g.inputs(): continue
            if sup[(P[path_cover.prev(v_m)],v_n)] <= L[path_cover.prev(v_m)]: #v_n < F.prev(v_m)
              if first == 'm': return False
              first = 'n'
            if sup[(P[path_cover.prev(v_n)],v_m)] <= L[path_cover.prev(v_n)]: #v_m < F.prev(v_n)
              if first == 'n': return False
              first = 'm'
    
    if update_graph: g.update_flow(successor_function)
    return True
  
def find_gadgets(g:BaseGraph[VT,ET]) -> Tuple[Dict[VT,VT],Dict[VT,Set[VT]]]:
  gadgets = dict()
  gadget_connections = dict()
  phases = g.phases()
  for v in g.vertices():
    if v not in g.inputs() and v not in g.outputs() and g.vertex_degree(v)==1:
      n = list(g.neighbors(v))[0]
      if not (g.type(v) == VertexType.Z and g.type(n) == VertexType.Z): continue
      if phases[n] not in (0,1): continue
      if n in gadgets: continue
      if n in g.inputs() or n in g.outputs(): continue
      gadgets[n] = v
      gadget_connections[n] = frozenset(set(g.neighbors(n)).difference({v}))
  return gadgets, gadget_connections

def remove_gadgets(g:BaseGraph[VT,ET], gadgets:Dict[VT,VT]) -> BaseGraph[VT,ET]:
  g_without = g.clone()
  g_without.remove_vertices(list(gadgets.keys())+list(gadgets.values()))
  return g_without

def fast_flow(
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
                return True
            return False
        else:
            processed |= Out_prime
            correctors = (correctors.difference(C_prime)) | (Out_prime.intersection(vertices.difference(inputs)))
            k += 1

class dipaths:
  def __init__(self, vertices:List[VT]) -> None:
    self.vertices = {v:False for v in vertices}
    self.arcs = {v:[[],[]] for v in vertices}
  def prev(self, v):
    return next(iter(self.arcs[v][0]),[])
  def next(self, v):
    return next(iter(self.arcs[v][1]),[])
  def add_arc(self, v, w):
    self.arcs[v][1].append(w)
    self.arcs[w][0].append(v)
    self.vertices[v] = True
    self.vertices[w] = True
  def del_arc(self, v, w):
    self.arcs[v][1].remove(w)
    if not self.arcs[v][0]: self.vertices[v] = False
    self.arcs[w][0].remove(v)
    if not self.arcs[w][1]: self.vertices[w] = False
          
def BuildPathCover(g:BaseGraph[VT, ET]) -> Optional[dipaths]:
  """Tries to build a path cover for g"""
  F = dipaths(g.vertices()) #collection of vertex disjoint dipaths in G
  visited = {v:0 for v in g.vertices()}
  i = 0
  for inp in g.inputs():
    i += 1
    F, visited, success = AugmentSearch(g, F, i, visited, inp)
    if not success: return None
  if len([v for v in g.vertices() if not F.vertices[v]]) == 0: return F
  else: return None

def AugmentSearch(g:BaseGraph[VT, ET], F:dipaths, iter:int, visited:Dict[VT,int], v:VT) -> Tuple[dipaths,Dict[VT,int],bool]:
  """Searches for an output vertex along pre-alternating walks for F starting at v, subject to limitations on the end-points of the search paths"""
  visited[v] = iter
  if v in g.outputs(): return(F, visited, True)
  if F.vertices[v] and v not in g.inputs() and visited[F.prev(v)] < iter:
    F, visited, success = AugmentSearch(g, F, iter, visited, F.prev(v))
    if success:
      F.del_arc(F.prev(v),v)
      return F, visited, True
  for w in g.neighbors(v):
    if visited[w] < iter and w not in g.inputs() and F.next(v) != w:
      if not F.vertices[w]:
        F, visited, success = AugmentSearch(g, F, iter, visited, w)
        if success:
          F.add_arc(v,w)
          return F, visited, True
      elif visited[F.prev(w)] < iter:
        F, visited, success = AugmentSearch(g, F, iter, visited, F.prev(w))
        if success:
          F.del_arc(F.prev(w),w)
          F.add_arc(v,w)
          return F, visited, True
  return F, visited, False

def GetChainDecomp(g:BaseGraph[VT, ET], C:dipaths) -> Tuple[Dict[VT,VT], Dict[VT,VT], Dict[VT,int]]:
  """Obtain the successor function f of the path cover C, and obtain functions describing the chain decomposition of the influencing digraph"""
  P = {v:None for v in g.vertices()}
  L = {v:0 for v in g.vertices()}
  f = {v:None for v in g.vertices() if v not in g.outputs()}
  for inp in g.inputs():
    l = 0
    v = inp
    while v not in g.outputs():
      try: f[v] = C.next(v)
      except: raise Exception(f'Vertex: {v}')
      P[v] = inp
      L[v] = l
      if C.next(v)==None: print(v)
      v = C.next(v)
      l += 1
    P[v] = inp
    L[v] = l
  return f, P, L

def ComputeSuprema(g:BaseGraph[VT, ET], f:Dict[VT,VT], P:Dict[VT,VT], L:Dict[VT,int]) -> Dict[Tuple[VT,VT],int]:
  """Compute the natural pre-order for successor function f in the form of a supremum function and functions characterising C"""
  sup, status = InitStatus(g,P,L)
  for v in [v for v in g.vertices() if v not in g.outputs()]:
    if not status[v]: sup, status = TraverseInflWalk(g,f,sup,status,v)
    if status[v] == 'pending': return False
  return sup
  
def InitStatus(g:BaseGraph[VT, ET], P:Dict[VT,VT], L:Dict[VT,int]):
  """Initialise the supremum function, and the status of each vertex"""
  sup = {(inp,v):None for v in g.vertices() for inp in g.inputs()}
  status = {v:None for v in g.vertices()}
  for v in g.vertices():
    for inp in g.inputs():
      if inp == P[v]: sup[(inp,v)]=L[v]
      else: sup[(inp,v)]=g.num_vertices()
    if v in g.outputs(): status[v]=True
  return sup, status

def TraverseInflWalk(g:BaseGraph[VT, ET], f:Dict[VT,VT], sup:Dict[Tuple[VT,VT],int], status:Dict[VT,bool], v:VT) -> Tuple[Dict[Tuple[VT,VT], Dict[VT,bool]]]:
  """Compute the suprema of v and all of it's descedants, by traversing influencing walks from v"""
  status[v] = 'pending'
  for w in list(g.neighbors(f[v]))+[f[v]]:
    if w != v:
      if not status[w]: sup, status = TraverseInflWalk(g,f,sup,status,w)
      if status[w] == 'pending': return sup, status
      else:
        for inp in g.inputs():
          if sup[(inp,v)] > sup[(inp,w)]: sup[(inp,v)] = sup[(inp,w)]
  status[v] = True
  return sup, status  