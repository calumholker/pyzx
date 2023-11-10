import sys, os
import time
from datetime import datetime
from numpy import NaN
import pandas as pd
import random
sys.path.append('../..')
import pyzx as zx
sys.path.append('/Users/calum/Developer/pyzx-heuristics-master')
import pyzx_heur as zx_heuristics # type: ignore

class Stats:
    def __init__(self, all_matches_df):
        self.all_matches_df = all_matches_df
        
    def add_match(self, match, rewrite, score, num_unfusions, iteration):
        idx = (rewrite, num_unfusions, score)
        if idx in self.all_matches_df.index:
            self.all_matches_df.loc[idx, 'total'] += 1
        else:
            data = {'total':1, 'flow_preserving':0, 'non_flow_preserving':0}
            self.all_matches_df.loc[idx] = data
        self.all_matches_df.sort_index(inplace=True)
        
    def match_flow(self, match, score, flow_preserving: bool):
        if match[0]:
            rewrite, num_unfusions = 'lcomp', len(match[0][2])
        elif match[1]:
            rewrite, num_unfusions = 'pivot', max(len(match[1][2][0]),len(match[1][2][1]))
        else: return
        
        idx = (rewrite, num_unfusions, score)
        if flow_preserving:
            self.all_matches_df.loc[idx, 'flow_preserving'] += 1
        else:
            self.all_matches_df.loc[idx, 'non_flow_preserving'] += 1

def generate_cliffordT_circuit(qubits: int, depth: int, p_cnot: float, p_t: float) -> zx.circuit.Circuit:
    p_s = 0.5*(1.0-p_cnot-p_t)
    p_had = 0.5*(1.0-p_cnot-p_t)
    c = zx.circuit.Circuit(qubits)
    for _ in range(depth):
        r = random.random()
        if r > 1-p_had: c.add_gate("HAD",random.randrange(qubits))
        elif r > 1-p_had-p_s: c.add_gate("S",random.randrange(qubits))
        elif r > 1-p_had-p_s-p_t: c.add_gate("T",random.randrange(qubits))
        else:
            tgt = random.randrange(qubits)
            while True:
                ctrl = random.randrange(qubits)
                if ctrl!=tgt: break
            c.add_gate("CNOT",tgt,ctrl)
    return c

def basic_optimise(c):
    c1 = zx.optimize.basic_optimization(c.copy(), do_swaps=False).to_basic_gates()
    c2 = zx.optimize.basic_optimization(c.copy(), do_swaps=True).to_basic_gates()
    if c2.twoqubitcount() < c1.twoqubitcount(): return c2
    return c1

def flow_opt(circuit, cFlow, max_lc_unfusions=0, max_p_unfusions=0, stats_obj=None):
    t0 = time.perf_counter()
    g = circuit.to_graph()
    zx.simplify.teleport_reduce(g)
    zx.simplify.to_graph_like(g, assert_bound_connections=False)
    zx.simplify.flow_2Q_simp(g, cFlow=cFlow, max_lc_unfusions=max_lc_unfusions, max_p_unfusions=max_p_unfusions, stats = stats_obj)
    if cFlow: c2 = zx.extract_simple(g,up_to_perm=False).to_basic_gates()
    else: c2 = zx.extract_circuit(g,up_to_perm=False).to_basic_gates()
    optimised_circuit = basic_optimise(c2)
    return optimised_circuit, time.perf_counter() - t0

def zx_heur_nu(circuit):
    c = zx_heuristics.Circuit.from_qc(circuit.to_qc())
    t0 = time.perf_counter()
    g = c.to_graph()
    g = zx_heuristics.simplify.teleport_reduce(g)
    g.track_phases = False
    zx_heuristics.simplify.greedy_simp_neighbors(g)
    c2 = zx_heuristics.extract_circuit(g, up_to_perm=False).to_basic_gates()
    optimised_circuit = basic_optimise(zx.Circuit.from_qc(c2.to_qc()))
    return optimised_circuit, time.perf_counter() - t0

def full_reduce(c):
    t0 = time.perf_counter()
    g = c.to_graph()
    zx.simplify.full_reduce(g,quiet=True)
    c2 = zx.extract_circuit(g,up_to_perm=False).to_basic_gates()
    return basic_optimise(c2), time.perf_counter() - t0
    
def result(strategy_id, strategy, rep, qubits, depth, pt, pcnot, initial_circ, opt_circ, optimisation_time, max_lc_unfusions=None, max_p_unfusions=None):
    return {
                'strategy_id': strategy_id,
                'strategy': strategy,
                'rep': rep,
                'max_lc_unfusions': max_lc_unfusions,
                'max_p_unfusions': max_p_unfusions,
                'qubits': qubits,
                'depth': depth,
                'pt': pt,
                'pcnot': pcnot,
                'initial_gate_count': len(initial_circ.gates),
                'initial_2Q_count': initial_circ.twoqubitcount(),
                'initial_T_count': initial_circ.tcount(),
                'final_gate_count': len(opt_circ.gates),
                'final_2Q_count': opt_circ.twoqubitcount(),
                'final_T_count': opt_circ.tcount(),
                'optimisation_time': optimisation_time
            }

if __name__ == "__main__":
    strategy_df = pd.DataFrame()
    strategy_id = 1
    
    all_matches_df = pd.DataFrame(columns=['total', 'flow_preserving', 'non_flow_preserving'], index=pd.MultiIndex.from_frame(pd.DataFrame(columns=['rewrite','num_unf','score']))).sort_index()
    
    circ_params = [(0, 10, 1000, 0.3, pt) for pt in [i * 0.01 for i in range(0, 16)]] + [(1, 10, d, 0.3, 0.1) for d in range(100, 2001, 100)]
    reps = 20
    
    for i in range(reps):
        for n, qubits, depth, pcnot, pt in circ_params:
            if n == 0 and i == 0 and pt < 0.06: continue
            results = []
            circuit = generate_cliffordT_circuit(qubits, depth, pcnot, pt)
            
            if n == 0:
                print(f"{depth}, {pt}, {i}, fr, {datetime.now().strftime('%H:%M:%S')}".ljust(100), end='\r', flush=True)
                fr_c, fr_t = full_reduce(circuit)
                fr_result = result(strategy_id, 'full_reduce', i, qubits, depth, pt, pcnot, circuit, fr_c, fr_t)
                results.append(fr_result)
                strategy_id += 1
                
                print(f"{depth}, {pt}, {i}, gF, {datetime.now().strftime('%H:%M:%S')}".ljust(100), end='\r', flush=True)
                gf_c, gf_t = flow_opt(circuit, cFlow=False)
                gf_result = result(strategy_id, 'gFlow', i, qubits, depth, pt, pcnot, circuit, gf_c, gf_t)
                results.append(gf_result)
                strategy_id += 1
                
                print(f"{depth}, {pt}, {i}, zxh, {datetime.now().strftime('%H:%M:%S')}".ljust(100), end='\r', flush=True)
                zxh_c, zxh_t = zx_heur_nu(circuit)
                zxh_result = result(strategy_id, 'ZX-heur', i, qubits, depth, pt, pcnot, circuit, zxh_c, zxh_t)
                results.append(zxh_result)
                strategy_id += 1
            
                for unf in range(6):
                    print(f"{depth}, {pt}, {i}, cF{unf}{unf}, {datetime.now().strftime('%H:%M:%S')}".ljust(100), end='\r', flush=True)
                    stats_obj = Stats(all_matches_df)
                    cf_c, cf_t = flow_opt(circuit, cFlow=True, max_p_unfusions=unf, max_lc_unfusions=unf, stats_obj=stats_obj)
                    cf_result = result(strategy_id, 'cFlow', i, qubits, depth, pt, pcnot, circuit, cf_c, cf_t, max_lc_unfusions=unf, max_p_unfusions=unf)
                    results.append(cf_result)
                    strategy_id += 1
                    all_matches_df = stats_obj.all_matches_df
            
            else:
                print(f"{depth}, {pt}, {i}, cF{2}{2}, {datetime.now().strftime('%H:%M:%S')}".ljust(100), end='\r', flush=True)
                stats_obj = Stats(all_matches_df)
                cf_c, cf_t = flow_opt(circuit, cFlow=True, max_p_unfusions=2, max_lc_unfusions=2, stats_obj=stats_obj)
                cf_result = result(strategy_id, 'cFlow', i, qubits, depth, pt, pcnot, circuit, cf_c, cf_t, max_lc_unfusions=2, max_p_unfusions=2)
                results.append(cf_result)
                strategy_id += 1
                all_matches_df = stats_obj.all_matches_df
            
            strategy_df = pd.concat([strategy_df, pd.DataFrame(results)])
            strategy_df.to_csv('strategy_df.csv')
            all_matches_df.to_csv('all_matches_df.csv')