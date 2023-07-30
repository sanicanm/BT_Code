# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 01:11:25 2023

@author: matth
"""

import numpy as np
import itertools as it
from Jaillet_Algorithm import Framework
from Efficient_Framework import EfficientFramework

n = 4
model = EfficientFramework(n)
model_compare = Framework(n) 

# Function to Calculate OPT
def FindMST(ordering):
    opt = []
    has_edge = [False for i in range(n-1)]
    base_spanned = False
    for edge in ordering:
        idx_hat = int((edge - 1) / 2)
        if edge == 0 and not base_spanned:
            opt.append(edge)
            base_spanned = True
        elif (not has_edge[idx_hat] or not base_spanned) and edge!=0:
            opt.append(edge)
            base_spanned = has_edge[idx_hat] or base_spanned
            has_edge[idx_hat] = True
        
    return opt



iters = 0
success_rate = 0
success_rate_c = 0
x = 0
y = 0

for total_ranks in it.permutations(range(0, model.m)):
    opt = FindMST(total_ranks)
    for size in range(0, model.m):
        for list_sampled_edges in it.combinations(total_ranks, size):
            iters += 1
            list_total_ranks = list(total_ranks)
            list_remaining_edges = np.setdiff1d(model.edge_indices, list_sampled_edges)
            list_remaining_edges = list_remaining_edges.tolist()
            has_edge = [False for i in range(model.n - 1)]
            
            next_list_picked_edges, list_selected_edges_ = model.Algorithm(list_total_ranks, list_sampled_edges, list_remaining_edges, has_edge)
            list_selected_edges_compare = model_compare.Algorithm(list_total_ranks, list_sampled_edges, list_remaining_edges)
          
            success_rate += 100 * len(set(list_selected_edges_)&set(opt))/len(opt)
            success_rate_c += 100 * len(set(list_selected_edges_compare)&set(opt))/len(opt)
            if len(set(list_selected_edges_)&set(opt)) != len(set(list_selected_edges_compare)&set(opt)):
                if len(set(list_selected_edges_)&set(opt)) > len(set(list_selected_edges_compare)&set(opt)):
                    y += 1
                else:
                    x += 1


print(success_rate/iters, "% successs rate")
print(success_rate_c/iters, "% successs rate")
print("Jaillet outperforms in", x/iters, "% of runs")
print("Entropy outperforms in", y/iters, "% of runs")