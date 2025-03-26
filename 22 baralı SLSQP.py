# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:06:19 2025

@author: Logzero
"""

import numpy as np
import time
from scipy.optimize import minimize

# 22 baralı, kayıpsız
tahmin = [10,10,10,10,10,12,10,10]

fun65 = lambda x: 0.0168*x[0]**2+7.0663*x[0]+6595.5+\
                  0.0127*x[1]**2+7.2592*x[1]+7290.6+\
                  0.0106*x[2]**2+5.682*x[2]+6780.5+\
                  0.0139*x[3]**2+3.1288*x[3]+1564.4+\
                  0.0168*x[4]**2+6.2232*x[4]+5134.1+\
                  0.021*x[5]**2+3.3128*x[5]+1159.5+\
                  0.0137*x[6]**2+3.2324*x[6]+1697+\
                  0.0147*x[7]**2+3.472*x[7]+1822.8

cons = ({'type': 'ineq', 'fun': lambda x: (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]) - 1503},  # non-negative olmalı
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] - 4000})  # toplamları 4000 ama sıfır olacak şekilde girmek lazım

bnds = ((190, 1120), (245, 1350), (318, 1432), (150, 600),
        (210, 990), (110, 420), (140, 630), (140, 630))

# Çözüm süresini ölçmek için başlangıç zamanı al
start_time = time.time()

res65 = minimize(fun65, tahmin, method='SLSQP', bounds=bnds, constraints=cons)

# Çözüm süresini hesapla
end_time = time.time()
elapsed_time = end_time - start_time

# Sonuçları yazdır
print("Problem 3:" " 22 Baralı Sistem")
print(f"Çözüm Süresi: {elapsed_time:.6f} saniye")
print("En iyi Değer:", res65.fun)
