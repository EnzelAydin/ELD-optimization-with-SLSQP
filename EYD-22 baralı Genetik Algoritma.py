# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:37:37 2025

@author: Logzero
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

# --------------------------------------------
# 
# --------------------------------------------
def objective_function_65(x):
    return (0.0168 * x[0]**2 + 7.0663 * x[0] + 6595.5 +
            0.0127 * x[1]**2 + 7.2592 * x[1] + 7290.6 +
            0.0106 * x[2]**2 + 5.682 * x[2] + 6780.5 +
            0.0139 * x[3]**2 + 3.1288 * x[3] + 1564.4 +
            0.0168 * x[4]**2 + 6.2232 * x[4] + 5134.1 +
            0.021 * x[5]**2 + 3.3128 * x[5] + 1159.5 +
            0.0137 * x[6]**2 + 3.2324 * x[6] + 1697 +
            0.0147 * x[7]**2 + 3.472 * x[7] + 1822.8)

# --------------------------------------------
# 
# --------------------------------------------
def penalized_objective_function(x):
    penalty = 0
    if abs(sum(x) - 4000) > 1e-3:  # Eşitlik kısıtı cezası
        penalty += 1e6 * abs(sum(x) - 4000)
    if sum(x) < 1503:  # Eşitsizlik kısıtı cezası
        penalty += 1e6 * (1503 - sum(x))
    return objective_function_65(x) + penalty

# --------------------------------------------
# 
# --------------------------------------------
def optimize_problem(bounds):
    start_time = time.time()
    result = differential_evolution(penalized_objective_function, bounds, strategy='best1bin', 
                                    maxiter=1000, popsize=400, mutation=(0.5, 1), recombination=0.7)
    end_time = time.time()
    
    return {
        "solution": result.x,
        "value": result.fun,
        "time": end_time - start_time
    }

# --------------------------------------------
# 
# --------------------------------------------
if __name__ == "__main__":
    bounds = [(190, 1120), (245, 1350), (318, 1432), (150, 600), (210, 990), (110, 420), (140, 630), (140, 630)]
    result_65 = optimize_problem(bounds)

    print("\nProblem 3: 22 Baralı Sistem")
    print(f"Çözüm Süresi: {result_65['time']:.2f} saniye")
    print(f"En İyi Değer: {result_65['value']:.2f}")
