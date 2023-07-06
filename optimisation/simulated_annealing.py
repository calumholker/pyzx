import sys
import os
import glob
import numpy as np
import random
import sqlite3
sys.path.append('/home/calum/pyzx')
import pyzx as zx
import concurrent.futures
from tqdm.auto import tqdm

evaluated_params = set()

def func(file, params):
    c = zx.Circuit.load(file).to_basic_gates()
    init_2Q = c.twoqubitcount()
    g = c.to_graph()
    zx.simplify.to_gh(g)
    g2 = zx.simplify.flow_reduce(g,x=params,quiet=True)
    c2 = zx.extract.extract_simple(g2, up_to_perm=False).to_basic_gates()
    c3 = zx.optimize.basic_optimization(c2).to_basic_gates()
    return c3.twoqubitcount()/init_2Q

def cost_function(params, files, tq=False):
    total = 0
    if tq:
        for file in tqdm(files, desc='Circuits'):
            total += func(file, params)
    else:
        for file in files:
            total += func(file, params)
    return total

def perturb_and_evaluate(params, files, perturbation_range, optimise_ratios):
    while True:
        new_params = np.copy(params)
        if optimise_ratios:
            new_params[:3] += np.random.uniform(-perturbation_range, perturbation_range, 3)
            new_params[:3] = np.clip(new_params[:3], 0, 1)
            if np.isclose(np.sum(new_params[:3]), 0): continue
            new_params[:3] /= np.sum(new_params[:3])
            new_params[:3] = np.round(new_params[:3], 2)
            new_params[2] = 1 - np.sum(new_params[:2])
        else:
            new_params[3:] += np.round(np.random.uniform(-perturbation_range*5, perturbation_range*5, 6), 2)
        params_tuple = tuple(new_params)
        if params_tuple not in evaluated_params:
            break
    new_cost = cost_function(new_params, files)
    evaluated_params.add(params_tuple)
    return new_params, new_cost

def simulated_annealing(files, init_params, temp=1000, cooling_rate=0.95, iter_limit=1000, num_perturbations=32, perturbation_range=0.3):
    current_params = np.array(init_params).astype(np.float64)
    current_cost = cost_function(current_params, files, tq=True)
    best_cost = current_cost
    best_params = [current_params]

    optimise_ratios = True
    no_improvement_count = 0
    switch_with_no_improvement = False

    for _ in tqdm(range(iter_limit), desc="Iterations"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(perturb_and_evaluate, current_params, files, perturbation_range, optimise_ratios) for _ in range(num_perturbations)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_perturbations, desc="Perturbations", leave=False):
                new_params, new_cost = future.result()
                save_to_database(new_params, temp, new_cost, optimise_ratios)
                if new_cost <= current_cost or np.exp((current_cost - new_cost) / temp) > np.random.rand():
                    current_params = new_params
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_params = [current_params]
                        no_improvement_count = 0
                        switch_with_no_improvement = False
                    elif current_cost == best_cost:
                        best_params.append(current_params)
                        no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else: no_improvement_count += 1

        if (optimise_ratios and no_improvement_count >= 256) or (not optimise_ratios and no_improvement_count >= 160):
            no_improvement_count = 0
            optimise_ratios = not optimise_ratios
            if switch_with_no_improvement: 
                current_params = random.choice(best_params)
                current_cost = best_cost
                switch_with_no_improvement = False
            else: switch_with_no_improvement = True

        temp *= cooling_rate
        perturbation_range -= 0.2
        perturbation_range *= cooling_rate**(1/2)
        perturbation_range += 0.2
    return current_params, current_cost

def save_to_database(params, temp, cost, optimise_ratios):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTOINCREMENT, p1 REAL, p2 REAL, p3 REAL, p4 REAL, p5 REAL, p6 REAL, p7 REAL, p8 REAL, p9 REAL, temp REAL, energy REAL, optimise_ratios INTEGER)''')
    c.execute("INSERT INTO results (p1, p2, p3, p4, p5, p6, p7, p8, p9, temp, energy, optimise_ratios) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*params, temp, cost, int(optimise_ratios)))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    directory = sys.argv[1]
    files = glob.glob(os.path.join(directory, "*"))
    init_params = [1/3,1/3,1/3,0,0,0,0,0,0]
    optimized_params, optimized_cost = simulated_annealing(files, init_params)
    print("Optimized Parameters:", optimized_params)
    print("Optimized Cost:", optimized_cost)