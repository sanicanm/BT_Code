# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 01:59:54 2023

@author: matth
"""

import time
import numpy as np
from numpy import random 
import math
from Efficient_Framework import EfficientFramework
from COMPLETED_FORMULA_VERSION import Framework


n = 4
model = EfficientFramework(n) 
formula = Framework(n) 

rng = np.random.default_rng()
total_ranks = rng.permutation(model.m)
#sample = rng.choice(total_ranks, math.ceil(model.m/2), replace=False, shuffle=False)
sample = np.arange(math.ceil(model.m/2)+1)
sample = np.delete(sample, 0)
unrevealed = np.setdiff1d(total_ranks, sample)

print(sample)

list_total_ranks = list(total_ranks)
list_revealed = list(sample)
list_unrevealed = list(unrevealed)

list_base_spanning_trees, list_non_base_spanning_trees = formula.GenerateSpanningTrees(list_revealed, list_unrevealed)


start_time = time.time()
H_2 = formula.CalculateEntropy(list_revealed, list_unrevealed, list_base_spanning_trees, list_non_base_spanning_trees)
end_time = time.time()
print(H_2)
print(f"Total Time Formula:{end_time -start_time}")

start_time = time.time()
H_1 = model.CalculateEntropy(list_revealed, list_unrevealed)
end_time = time.time()
print(H_1)
print(f"Total Time Efficient:{end_time -start_time}")


