# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:38:56 2023

@author: matth
"""
import itertools as it
from itertools import combinations, permutations

def interleave(m, non_sample_size, interleaving, sample, non_sample):
    all_w = [0 for i in range(m)]
    idx_interleaving, idx_sample, idx_non_sample = 0, 0, 0
    assert(non_sample_size + len(sample) == m)
    for i in range(m):
        if idx_interleaving < non_sample_size and i == interleaving[idx_interleaving]:
            all_w[i] = non_sample[idx_non_sample]
            idx_interleaving += 1
            idx_non_sample += 1
        else:
            #print("sample index is", idx_sample, "non sample index is", idx_non_sample, "i is", i)
            all_w[i] = sample[idx_sample]
            idx_sample += 1
            #print("NEW sample index is", idx_sample)
    
    return all_w


def LinearExtension(elements_to_sort, sampling_partial_order, list_of_pair_orders_one, list_of_pair_orders_two):
    crazy_number = 0
    non_sample = list(set(elements_to_sort) - set(sampling_partial_order))
    #print("sample", sampling_partial_order, "non sample", non_sample, "m=", len(elements_to_sort))
    for non_sample_w in it.permutations(non_sample):
        for interleaving in it.combinations(range(0,len(elements_to_sort)), len(non_sample)):
            all_w = interleave(len(elements_to_sort), len(non_sample), interleaving, sampling_partial_order, non_sample_w)
            violation = False
            for (j,k) in list_of_pair_orders_one:
                if all_w.index(k) < all_w.index(j):
                    violation = True
                    break
            if violation == False:
                for (j,k) in list_of_pair_orders_two:
                    if all_w.index(k) < all_w.index(j):
                        violation = True
                        break
            if violation == False:
                crazy_number += 1
            
    return crazy_number


l_1 = [1,2,3,4]
l_2 = [1,2,4]
r_1 = [(3,4)]
r_2 = []
#print(LinearExtension(l_1,l_2,r_1,r_2))


"""
def LinearExtensionFast(elements_to_sort, sampling_partial_order, list_of_pair_orders):
    dict_of_intervals = {}
    number_extensions = 1
    for (e_1, e_2) in list_of_pair_orders:
        if e_2 in sampling_partial_order:
            temp_set = []
            for i in range(0, sampling_partial_order.index(e_2) + 1):
                temp_set.append(i)
            number_extensions *= len(temp_set)
            dict_of_intervals[e_1] = temp_set
            print("Temp dict greater:",dict_of_intervals)
        else:
            temp_set = []
            for i in range(sampling_partial_order.index(e_1) + 2, len(sampling_partial_order) + 2):
                temp_set.append(i)
            number_extensions *= len(temp_set)
            dict_of_intervals[e_2] = temp_set
            print("Temp dict lesser:",dict_of_intervals)
    print(dict_of_intervals)

#print(LinearExtensionFast(l_1,l_2,r_1,r_2))


"""
    
    
    
    
