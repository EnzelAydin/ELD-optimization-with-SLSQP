# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 23:45:05 2025

@author: enzel
"""

import numpy as np
import time
from collections import deque

# --------------------------------------------
# Problem Tanımı
# --------------------------------------------
def objective_function_65(x):
    return (0.0106 * x[0]**2 + 5.682 * x[0] + 6780.5 +
            0.0139 * x[1]**2 + 3.1288 * x[1] + 1564.4 +
            0.0168 * x[2]**2 + 6.2232 * x[2] + 5134.1 +
            0.021 * x[3]**2 + 3.3128 * x[3] + 1159.5 +
            0.0137 * x[4]**2 + 3.2324 * x[4] + 1697 +
            0.0147 * x[5]**2 + 3.472 * x[5] + 1822.8)

# --------------------------------------------
# Kısıtları Ceza Fonksiyonu Olarak Dahil Etme
# --------------------------------------------
def penalized_objective_function(x):
    penalty = 0
    constraint_sum = sum(x)
    if abs(constraint_sum - 2734.9) > 1e-3:
        penalty += 1e6 * abs(constraint_sum - 2734.9)
    if constraint_sum < 1068:
        penalty += 1e6 * (1068 - constraint_sum)
    return objective_function_65(x) + penalty

# --------------------------------------------
# Geliştirilmiş Tabu Arama Optimizasyonu
# --------------------------------------------
def optimize_problem(bounds, max_iterations=30000, tabu_size=1000, initial_step=0.3, min_step=0.005):
    start_time = time.time()
    
    # Başlangıç çözümü rastgele seçilir
    current_solution = np.array([np.random.uniform(low, high) for (low, high) in bounds])
    best_solution = current_solution.copy()
    best_value = penalized_objective_function(best_solution)
    
    tabu_list = deque(maxlen=tabu_size)
    step_factor = initial_step
    stagnation_counter = 0
    
    for iteration in range(max_iterations):
        neighbors = []
        step_size = step_factor * (np.array([high - low for (low, high) in bounds]))
        
        for _ in range(50):  # Daha fazla komşu üret
            new_solution = current_solution.copy()
            index = np.random.randint(len(bounds))
            new_solution[index] += np.random.uniform(-step_size[index], step_size[index])
            new_solution[index] = np.clip(new_solution[index], bounds[index][0], bounds[index][1])
            if tuple(new_solution) not in tabu_list:
                neighbors.append(new_solution)
        
        if not neighbors:
            continue
        
        new_solution = min(neighbors, key=penalized_objective_function)
        new_value = penalized_objective_function(new_solution)
        
        if new_value < best_value:
            best_solution = new_solution.copy()
            best_value = new_value
            step_factor = initial_step  # Adım boyutunu sıfırla
            stagnation_counter = 0  # Durgunluk sayacını sıfırla
        else:
            step_factor = max(step_factor * 0.98, min_step)  # Adım boyutunu küçült
            stagnation_counter += 1
        
        if stagnation_counter > 500:  # Çok uzun süre iyileşme olmazsa rastgele sıçrama yap
            current_solution = np.array([np.random.uniform(low, high) for (low, high) in bounds])
            stagnation_counter = 0
        else:
            current_solution = new_solution
            tabu_list.append(tuple(new_solution))
    
    end_time = time.time()
    return {
        "solution": best_solution,
        "value": best_value,
        "time": end_time - start_time
    }

# --------------------------------------------
# Problemi Çöz
# --------------------------------------------
if __name__ == "__main__":
    bounds = [(318, 1432), (150, 600), (210, 990), (110, 420), (140, 630), (140, 630)]
    result_65 = optimize_problem(bounds)

    print("\nProblem 3: 14 Baralı Sistem")
    print(f"Çözüm Süresi: {result_65['time']:.2f} saniye")
    print(f"En İyi Değer: {result_65['value']:.2f}")