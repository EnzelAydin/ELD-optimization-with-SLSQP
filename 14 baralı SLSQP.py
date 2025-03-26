# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:01:22 2025

@author: Logzero
"""

import numpy as np
import time
from scipy.optimize import minimize

# 14 baralı 

tahmin = [10,10,10,10,10,12]

fun43 = lambda x: 0.0106*x[0]**2+5.682*x[0]+6780.5+\
                  0.0139*x[1]**2+3.1288*x[1]+1564.4+\
                  0.0168*x[2]**2+6.2232*x[2]+5134.1+\
                  0.021*x[3]**2+3.3128*x[3]+1159.5+\
                  0.0137*x[4]**2+3.2324*x[4]+1697+\
                  0.0147*x[5]**2+3.472*x[5]+1822.8
cons = ({'type': 'ineq', 'fun': lambda x:  (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])-1068},# non-negative olmali,
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] -2734.9})#toplamlari 800 ama sifir olacak sekilde girmek lazim
bnds = ((318, 1432),(150, 600),(210, 990),(110, 420),(140, 630),(140, 630))

# Çözüm süresini ölçmek için başlangıç zamanı al
start_time = time.time()

res42 = minimize(fun43, tahmin, method='SLSQP', bounds=bnds,constraints=cons)

# Çözüm süresini hesapla
end_time = time.time()
elapsed_time = end_time - start_time

# Sonuçları yazdır
print("Problem 3:" " 14 Baralı Sistem")
print(f"Çözüm Süresi: {elapsed_time:.6f} saniye")
print("En iyi Değer:", res42.fun)