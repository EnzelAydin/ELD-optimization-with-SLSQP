# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 02:48:25 2025

@author: Logzero
"""

import numpy as np
import time

# --------------------------------------------
# Problem Tanımı
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
# Kısıtları Ceza Fonksiyonu Olarak Dahil Etme
# --------------------------------------------
def penalized_objective_function(x):
    penalty = 0
    constraint_sum = sum(x)
    # Ceza katsayısı; daha iyi hassasiyet için azaltıldı
    if abs(constraint_sum - 4000) > 1e-3:
        penalty += 1e3 * abs(constraint_sum - 4000)
    if constraint_sum < 1503:
        penalty += 1e3 * (1503 - constraint_sum)
    return objective_function_65(x) + penalty

# --------------------------------------------
# Yapay Arı Kolonisi Optimizasyonu (İyileştirilmiş)
# --------------------------------------------
def optimize_problem(bounds, colony_size=400, max_cycles=5000, limit=50):
    start_time = time.time()
    num_params = len(bounds)
    # Başlangıçta rastgele oluşturulan popülasyon
    population = np.array([[np.random.uniform(low, high) for (low, high) in bounds] for _ in range(colony_size)])
    fitness = np.array([1 / (1 + penalized_objective_function(ind)) for ind in population])
    trial_counters = np.zeros(colony_size)
    
    initial_local_radius = 0.02  # Yerel arama başlangıç yarıçapı
    scout_radius = 0.05         # Scout (yeniden başlatma) yarıçapı yakınlık parametresi
    
    for cycle in range(max_cycles):
        # Adaptif yerel arama yarıçapı: iterasyon ilerledikçe daha hassas arama
        local_search_radius = initial_local_radius * (1 - cycle / max_cycles)
        
        for i in range(colony_size):
            # Yeni çözüm için küçük bir mutasyon (phi) uygulanıyor
            phi = np.random.uniform(-0.5, 0.5, num_params)
            partner = np.random.randint(0, colony_size)
            while partner == i:
                partner = np.random.randint(0, colony_size)
            new_solution = population[i] + phi * (population[i] - population[partner])
            new_solution = np.clip(new_solution, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fitness = 1 / (1 + penalized_objective_function(new_solution))
            
            if new_fitness > fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        best_index = np.argmax(fitness)
        best_solution = population[best_index]
        best_value = penalized_objective_function(best_solution)
        
        # Adaptif yerel arama: en iyi çözümün etrafında küçük değişiklikler yaparak ince ayar
        local_solution = best_solution + np.random.uniform(-local_search_radius, local_search_radius, num_params)
        local_solution = np.clip(local_solution, [b[0] for b in bounds], [b[1] for b in bounds])
        local_value = penalized_objective_function(local_solution)
        if local_value < best_value:
            best_solution = local_solution
            best_value = local_value

        # Akıllı scout: Eğer belirli bir sayının üzerinde başarısızlık varsa, en kötü birey
        # en iyi çözüme yakın bir bölgeden yeniden başlatılır
        if np.any(trial_counters > limit):
            worst_index = np.argmax(trial_counters)
            population[worst_index] = best_solution + np.random.uniform(-scout_radius, scout_radius, num_params)
            population[worst_index] = np.clip(population[worst_index], [b[0] for b in bounds], [b[1] for b in bounds])
            fitness[worst_index] = 1 / (1 + penalized_objective_function(population[worst_index]))
            trial_counters[worst_index] = 0
    
    end_time = time.time()
    return {
        "solution": best_solution,
        "value": best_value,
        "time": end_time - start_time
    }

# --------------------------------------------
# Ek Test Durumu (Farklı Rastgele Başlangıçlarla Çalıştırma)
# --------------------------------------------
def run_tests():
    bounds = [(190, 1120), (245, 1350), (318, 1432), (150, 600),
              (210, 990), (110, 420), (140, 630), (140, 630)]
    
    # Standart test
    result = optimize_problem(bounds)
    print("\nTest 1 - Standart Çalıştırma")
    print(f"Çözüm Süresi: {result['time']:.2f} saniye")
    print(f"En İyi Değer: {result['value']:.2f}")
    
    # Ek test: Farklı kolon büyüklüğü
    result2 = optimize_problem(bounds, colony_size=300)
    print("\nTest 2 - Koloni Boyutu 300")
    print(f"Çözüm Süresi: {result2['time']:.2f} saniye")
    print(f"En İyi Değer: {result2['value']:.2f}")
    
    # Ek test: Farklı iterasyon sayısı
    result3 = optimize_problem(bounds, max_cycles=6000)
    print("\nTest 3 - Maksimum Döngü 6000")
    print(f"Çözüm Süresi: {result3['time']:.2f} saniye")
    print(f"En İyi Değer: {result3['value']:.2f}")

# --------------------------------------------
# Problemi Çöz
# --------------------------------------------
if __name__ == "__main__":
    run_tests()
