# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:09:54 2023

@author: matth
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx import create_empty_copy
from itertools import combinations
from itertools import permutations
import copy
import itertools as it
import numpy as np
import  math
from Crazy_Number import LinearExtension


class HatUsed():
    def __init__(self, sequence_, edge_l_, edge_h_ = None):
        self.edge_l = edge_l_ 
        self.edge_h = edge_h_
        self.sequence = sequence_
    fully_rev = False
    half_rev = False
    unrev = False
        

class Framework():
    def __init__(self, nodes):
        self.n = nodes
        self.g = nx.Graph()
        self.g.add_node(1, pos = (0,0))
        self.g.add_node(2, pos = (self.n*2,0))
        self.g.add_edge(1,2, index = 0)
        
        
        temp_iter = 1
        for i in range (3,self.n+1):
            self.g.add_node(i, pos = (self.n,i))
            self.g.add_edge(1,i, index = temp_iter)
            temp_iter += 1
            self.g.add_edge(2,i, index = temp_iter)
            temp_iter += 1
        self.edge_number = temp_iter # replace by self.m at some point
        self.m = temp_iter
        del temp_iter
        self.edge_indices = np.arange(0, self.edge_number)


        
        
    def DrawGraph(self, g, title):
        edge_labels = nx.get_edge_attributes(g, "weight")
        pos = nx.get_node_attributes(self.g,'pos')
        nx.draw_networkx_nodes(g, pos, node_size=10)
        nx.draw_networkx_edges(g, pos, g.edges())
        #nx.draw_networkx_labels(g, self.pos,font_size=8)
        nx.draw_networkx_edge_labels(g, pos, edge_labels,font_size=8)
        plt.title(title)
        
    def PlotGraph(self, g, title = ''):
        self.DrawGraph(g, title)
        plt.show()
    """
    def PickEdges(self): # look at pre defined weights later on
        list_picked_edges = []
        list_remaining_edges = []
        
        #Create an emtpy copy of self.g to see what the sample looks like
        temp_graph = create_empty_copy(self.g)
        
        sample_p = 0.5
        random_seed = 42 
        rng_sample = np.random.default_rng()
        is_sampled = rng_sample.binomial(1, sample_p, self.m)
        
        #Pick edges from self.g
        for (u, v) in self.g.edges():
            if is_sampled[self.g.edges[u,v]['index']] and (u,v) != (1,2): #and (u,v) != (1,2):
                list_picked_edges.append(self.g.edges[u,v]['index'])
                temp_graph.add_edge(u, v)
            else:
                list_remaining_edges.append(self.g.edges[u,v]['index'])
                
        random.shuffle(list_picked_edges)
        #Plot the picked edges
        self.PlotGraph(temp_graph, 'sampled edges')
        print("Number of sampled edges is: ", len(list_picked_edges))
        print("Edges are (ranked):", list_picked_edges)
        del temp_graph
                
        return list_picked_edges, list_remaining_edges
    """
    
    def GenerateSpanningTrees(self, list_revealed_edges, list_unrevealed_edges):
        list_base_spanning_trees = []
        list_non_base_spanning_trees = []
        print_list_non_base_spanning_trees = []
        sampled_hat_indices_dict = {}
        opt_edges = []
        base_spanned_sample = False
        lighter_edge_max_hat = None
        
        candidate_edges = list(self.edge_indices)
        candidate_edges.remove(0)
        candidate_hats = list(candidate_edges)
        
        revealed_hats = list(list_revealed_edges)
        #if 0 in revealed_hats:
        #    revealed_hats.remove(0)
        base_out = False
        for edge in revealed_hats:
            if edge == 0 and base_spanned_sample == False:
                base_spanned_sample = True
                continue
            elif edge == 0 and base_spanned_sample == True:
                base_out = True
                continue
                
            hat_index = int((edge-1)/2)
            if hat_index in sampled_hat_indices_dict.keys() and base_spanned_sample == False:
                base_spanned_sample = True
                opt_edges.append(sampled_hat_indices_dict[hat_index])
                lighter_edge_max_hat = edge
                
                candidate_edges.remove(sampled_hat_indices_dict[hat_index])
                candidate_edges.remove(edge)
                candidate_hats.remove(sampled_hat_indices_dict[hat_index])
                candidate_hats.remove(edge)
            elif hat_index in sampled_hat_indices_dict.keys() and base_spanned_sample == True:
                opt_edges.append(sampled_hat_indices_dict[hat_index])
                
                candidate_edges.remove(sampled_hat_indices_dict[hat_index])
                candidate_edges.remove(edge)
                
                if sampled_hat_indices_dict[hat_index] in candidate_hats:
                    candidate_hats.remove(sampled_hat_indices_dict[hat_index])
                if edge in candidate_hats:
                    candidate_hats.remove(edge)
            elif hat_index not in sampled_hat_indices_dict.keys():
                sampled_hat_indices_dict[hat_index] = edge
                if base_spanned_sample == True:
                    candidate_hats.remove(edge)
                    candidate_hats.remove(edge - (-1)**(edge % 2))

        
        #print("Candidate edges are:", candidate_edges)
        #print("Opt edges are,", opt_edges)
        initial_boolean_tree_array = [0 for i in self.edge_indices]
        for j in opt_edges:
            initial_boolean_tree_array[j] = 1
            
        #base trees
        #print("base candidate edges are:", base_candidate_edges, "lenght is:", len(base_candidate_edges))
        if base_out == False:
            for binary_sequence in it.product([0, 1], repeat=int(len(candidate_edges)/2)):
                
                temp_tree_array = list(initial_boolean_tree_array)
                temp_tree_array[0] = 1
                for i in range(0, int(len(candidate_edges)/2)):
                    if binary_sequence[i]==1: 
                        temp_tree_array[candidate_edges[2*i]] = 1
                    else: 
                        temp_tree_array[candidate_edges[2*i + 1]] = 1
                        
                temp_feas = True
                for edge_index in range(0,len(temp_tree_array)):
                    if temp_tree_array[edge_index] == 1 and (edge_index - (-1)**(edge_index % 2)) in list_revealed_edges  and 0 in list_revealed_edges:
                        if list_revealed_edges.index(edge_index - (-1)**(edge_index % 2)) < list_revealed_edges.index(0):
                            temp_feas = False
                if temp_feas == True: 
                    list_base_spanning_trees.append(temp_tree_array)
                del temp_tree_array
        #print(list_base_spanning_trees)
        #print("number of possible base spanning trees is:",len(list_base_spanning_trees))
        
        #non base trees
        for i in range(0, int(len(candidate_hats)/2)):
            temp_candidate_edges = list(candidate_edges)
            temp_candidate_edges.remove(candidate_hats[2*i])
            temp_candidate_edges.remove(candidate_hats[2*i + 1])
            
            
            for binary_sequence in it.product([0, 1], repeat=int(len(temp_candidate_edges)/2)):
                temp_tree_array = list(initial_boolean_tree_array)
                temp_tree_array[candidate_hats[2*i]] = 1
                temp_tree_array[candidate_hats[2*i + 1]] = 1
                
                revealed_are = ('both' if (candidate_hats[2*i] in list_revealed_edges and candidate_hats[2*i + 1] in list_revealed_edges) else( '1st' if (candidate_hats[2*i] in list_revealed_edges) else '2nd'))
                temp_flag = (candidate_hats[2*i] if candidate_hats[2*i] in list_revealed_edges else candidate_hats[2*i + 1])
                if revealed_are == 'both':
                    if list_revealed_edges.index(candidate_hats[2*i]) < list_revealed_edges.index(candidate_hats[2*i + 1]):
                        temp_flag = candidate_hats[2*i + 1]
                    else:
                        temp_flag = candidate_hats[2*i]
                elif revealed_are == '1st': 
                    temp_flag = candidate_hats[2*i]
                else: 
                    assert(revealed_are == '2nd')
                    temp_flag = candidate_hats[2*i + 1]
                for j in range(0, int(len(temp_candidate_edges)/2)):
                    if binary_sequence[j]==1: 
                        temp_tree_array[temp_candidate_edges[2*j]] = 1
                    else: 
                        temp_tree_array[temp_candidate_edges[2*j + 1]] = 1
                        
                if candidate_hats[2*i] in list_revealed_edges:
                    temp_tree_class = HatUsed(temp_tree_array, candidate_hats[2*i + 1], candidate_hats[2*i])
                    temp_tree_class.half_rev = True
                elif candidate_hats[2*i + 1] in list_revealed_edges:
                    temp_tree_class = HatUsed(temp_tree_array, candidate_hats[2*i], candidate_hats[2*i + 1])
                    temp_tree_class.half_rev = True
                else: 
                    temp_tree_class = HatUsed(temp_tree_array, candidate_hats[2*i], candidate_hats[2*i + 1])
                    temp_tree_class.unrev = True
                
                temp_feas = True
                for edge_index in range(0,len(temp_tree_array)):
                    if (edge_index != candidate_hats[2*i]) and (edge_index != candidate_hats[2*i+1]):
                        if temp_tree_array[edge_index] == 1 and (edge_index - (-1)**(edge_index % 2)) in list_revealed_edges  and temp_flag in list_revealed_edges:
                            if list_revealed_edges.index(edge_index - (-1)**(edge_index % 2)) < list_revealed_edges.index(temp_flag):
                                temp_feas = False
                if temp_feas == True:          
                    list_non_base_spanning_trees.append(temp_tree_class)
                    print_list_non_base_spanning_trees.append(temp_tree_array)
                
                del temp_tree_array, temp_tree_class
            del temp_candidate_edges
        
        if type(lighter_edge_max_hat) == int :
            for binary_sequence in it.product([0, 1], repeat=int(len(candidate_edges)/2)):
                temp_tree_array = list(initial_boolean_tree_array)
                temp_tree_array[lighter_edge_max_hat] = 1
                for i in range(0, int(len(candidate_edges)/2)):
                    if binary_sequence[i]==1: 
                        temp_tree_array[candidate_edges[2*i]] = 1
                    else: 
                        temp_tree_array[candidate_edges[2*i + 1]] = 1
                
                temp_feas = True
                for edge_index in range(0,len(temp_tree_array)):
                    if (edge_index != lighter_edge_max_hat) and (edge_index != lighter_edge_max_hat - (-1)**(lighter_edge_max_hat % 2)):
                        if temp_tree_array[edge_index] == 1 and (edge_index - (-1)**(edge_index % 2)) in list_revealed_edges:
                            if list_revealed_edges.index(edge_index - (-1)**(edge_index % 2)) < list_revealed_edges.index(lighter_edge_max_hat):
                                temp_feas = False
                            
                temp_tree_class = HatUsed(temp_tree_array, lighter_edge_max_hat)
                temp_tree_class.fully_rev = True
                
                if temp_feas == True:    
                    list_non_base_spanning_trees.append(temp_tree_class)
                    print_list_non_base_spanning_trees.append(temp_tree_array)
                del temp_tree_array, temp_tree_class
            
        #print(list_non_base_spanning_trees)
        #print("number of possible non base spanning trees is:",len(list_non_base_spanning_trees))
        #print("Number of trees to go through is:", len(list_non_base_spanning_trees) + len(list_base_spanning_trees), "Number of base trees to go through is:",  len(list_base_spanning_trees),"Number of non base trees to go through is:", len(list_non_base_spanning_trees))
        #print("Non base trees:", print_list_non_base_spanning_trees)
        #print("base trees:", list_base_spanning_trees)
        
        return list_base_spanning_trees, list_non_base_spanning_trees
        
    def CalculateEntropy(self, list_revealed_edges, list_unrevealed_edges, list_base_spanning_trees, list_non_base_spanning_trees):
        spanning_tree_occurrences_all = []
        for spanning_tree_class in list_non_base_spanning_trees:
            spanning_tree_occurrences_single = 0
            spanning_tree = spanning_tree_class.sequence 
            if spanning_tree_class.fully_rev == True: #Implement if hat is fully revealed
                Z = []
                y_1 = []
                y_2_selected = []
                y_2_unselected = []
                ranked_pairs_one = []
                remaining_edges = list(list_unrevealed_edges)
                
                non_base_edges = list(list_revealed_edges)
                #if 0 in non_base_edges:
                #    non_base_edges.remove(0)
                sampling_po = list(list_revealed_edges)
                sampling_po.remove(spanning_tree_class.edge_l)
                
                
                for edge in non_base_edges:
                    if spanning_tree[edge] == 1 and list_revealed_edges.index(edge) < list_revealed_edges.index(spanning_tree_class.edge_l):
                        y_1.append(edge)
                        sampling_po.remove(edge)
                    elif spanning_tree[edge] == 1 and list_revealed_edges.index(edge) > list_revealed_edges.index(spanning_tree_class.edge_l):
                        y_2_selected.append(edge)
                        if edge - (-1)**(edge % 2) in list_unrevealed_edges:
                            ranked_pairs_one.append((edge, edge - (-1)**(edge % 2)))
                            assert(spanning_tree[edge - (-1)**(edge % 2)] == 0)
                    elif edge != spanning_tree_class.edge_l and edge != spanning_tree_class.edge_h:
                        y_2_unselected.append(edge)
                for edge in list_unrevealed_edges:
                    if spanning_tree[edge] == 1:
                        Z.append(edge)
                        remaining_edges.remove(edge)
                
                for binary_sequence in it.product([0, 1], repeat=len(Z)):
                    #print("Binary Sequence:", binary_sequence)
                    ranked_pairs_two = []
                    z_1 = []
                    z_2 = []
                    two_prime = 0
                    for i in range (0,len(binary_sequence)):
                        if binary_sequence[i] == 1:
                            z_1.append(Z[i])
                        else:
                            z_2.append(Z[i])
                            if Z[i] - (-1)**(Z[i] % 2) in list_revealed_edges:
                                ranked_pairs_two.append((Z[i], Z[i] - (-1)**(Z[i] % 2)))
                            else:
                                two_prime += 1
                    
                    #sampling_po = list(y_2_selected)
                    #sampling_po.extend(y_2_unselected)
                    elements_to_sort = list(sampling_po)
                    elements_to_sort.extend(z_2)
                    elements_to_sort.extend(remaining_edges)
                    #print("Z is", Z,"elements_to_sort:", elements_to_sort, "partial order is:",sampling_po, "ranking 1,2:", ranked_pairs_one, ranked_pairs_two)
                    if len(z_1) == 0:
                        easy_number = 1
                    else:
                        easy_number =  math.factorial(len(y_1)+len(z_1))/math.factorial(len(y_1)) #* math.factorial(len(Z))/math.factorial(len(Z)-len(z_1)) 
                    crazy_number = LinearExtension(elements_to_sort, sampling_po, ranked_pairs_one, ranked_pairs_two)
                    spanning_tree_occurrences_single += easy_number*crazy_number/(2**(two_prime))
                    #print("easy and crazy are:",easy_number,crazy_number, "spanning tree occ:",spanning_tree_occurrences_single)
                    
                spanning_tree_occurrences_all.append(spanning_tree_occurrences_single)
                
            elif spanning_tree_class.half_rev == True: # The hat is half revealed
                 temp_index = 0
                 for edge in list_revealed_edges:
                     if spanning_tree[edge] == 1:
                         assert(list_revealed_edges.index(edge) == temp_index)
                         temp_index += 1
                     else:
                         break
                 for r in range (1,temp_index+2):
                     temp_list_revealed_edges, temp_list_unrevealed_edges = self.AddEdgeToSample(spanning_tree_class.edge_l, r, list_revealed_edges, list_unrevealed_edges)
                     Z = []
                     y_1 = []
                     y_2_selected = []
                     y_2_unselected = []
                     ranked_pairs_one = []
                     remaining_edges = list(temp_list_unrevealed_edges)
                     
                     non_base_edges = list(temp_list_revealed_edges)
                     #if 0 in non_base_edges: 
                     #    non_base_edges.remove(0)
                         
                     flag_edge = (spanning_tree_class.edge_l if (temp_list_revealed_edges.index(spanning_tree_class.edge_h) < temp_list_revealed_edges.index(spanning_tree_class.edge_l)) else spanning_tree_class.edge_h )
                     sampling_po = list(temp_list_revealed_edges)
                     sampling_po.remove(flag_edge)
                     
                     for edge in non_base_edges:
                         if spanning_tree[edge] == 1 and temp_list_revealed_edges.index(edge) < temp_list_revealed_edges.index(flag_edge):
                             y_1.append(edge)
                             sampling_po.remove(edge)
                         elif spanning_tree[edge] ==1 and temp_list_revealed_edges.index(edge) > temp_list_revealed_edges.index(flag_edge):
                             y_2_selected.append(edge)
                             if edge - (-1)**(edge % 2) in temp_list_unrevealed_edges:
                                 ranked_pairs_one.append((edge, edge - (-1)**(edge % 2)))
                                 assert(spanning_tree[edge - (-1)**(edge % 2)] == 0)
                         elif edge != spanning_tree_class.edge_l and edge != spanning_tree_class.edge_h:
                             y_2_unselected.append(edge)
                     
                     for edge in temp_list_unrevealed_edges:
                         if spanning_tree[edge] == 1:
                             Z.append(edge)
                             remaining_edges.remove(edge)
                     
                     
                     for binary_sequence in it.product([0, 1], repeat=len(Z)):
                         #print("Binary Sequence:", binary_sequence)
                         ranked_pairs_two = []
                         z_1 = []
                         z_2 = []
                         two_prime = 0
                         for i in range (0,len(binary_sequence)):
                             if binary_sequence[i] == 1:
                                 z_1.append(Z[i])
                             else:
                                 z_2.append(Z[i])
                                 if Z[i] - (-1)**(Z[i] % 2) in list_revealed_edges:
                                     ranked_pairs_two.append((Z[i], Z[i] - (-1)**(Z[i] % 2)))
                                 else:
                                     two_prime += 1
                         
                         #sampling_po = list(y_2_selected)
                         #sampling_po.extend(y_2_unselected)
                         elements_to_sort = list(sampling_po)
                         elements_to_sort.extend(z_2)
                         elements_to_sort.extend(remaining_edges)
                         #print("Z is", Z,"elements_to_sort:", elements_to_sort, "partial order is:",sampling_po, "ranking 1,2:", ranked_pairs_one, ranked_pairs_two)
                         if len(z_1) == 0:
                             easy_number = 1
                         else:
                             easy_number =  math.factorial(len(y_1)+len(z_1))/math.factorial(len(y_1)) #* math.factorial(len(Z))/math.factorial(len(Z)-len(z_1)) 
                         crazy_number = LinearExtension(elements_to_sort, sampling_po, ranked_pairs_one, ranked_pairs_two)
                         spanning_tree_occurrences_single += easy_number*crazy_number/(2**(two_prime))
                         #print("easy and crazy are:",easy_number,crazy_number, "spanning tree occ:",spanning_tree_occurrences_single)
                         
                 spanning_tree_occurrences_all.append(spanning_tree_occurrences_single)
            else: # The hat is  unrevealed
                 assert(spanning_tree_class.unrev == True)
                 temp_index = 0
                 for edge in list_revealed_edges:
                     if spanning_tree[edge] == 1:
                         assert(list_revealed_edges.index(edge) == temp_index)
                         temp_index += 1
                     else:
                         break
                 for r_1 in range (1,temp_index+2):
                     first_temp_list_revealed_edges, first_temp_list_unrevealed_edges = self.AddEdgeToSample(spanning_tree_class.edge_l, r_1, list_revealed_edges, list_unrevealed_edges)
                     for r_2 in range(1,temp_index+3):
                         temp_list_revealed_edges, temp_list_unrevealed_edges = self.AddEdgeToSample(spanning_tree_class.edge_h, r_2, first_temp_list_revealed_edges, first_temp_list_unrevealed_edges)
                         Z = []
                         y_1 = []
                         y_2_selected = []
                         y_2_unselected = []
                         ranked_pairs_one = []
                         remaining_edges = list(temp_list_unrevealed_edges)
                         
                         non_base_edges = list(temp_list_revealed_edges)
                         #if 0 in non_base_edges: 
                         #    non_base_edges.remove(0)
                             
                         flag_edge = (spanning_tree_class.edge_l if (temp_list_revealed_edges.index(spanning_tree_class.edge_h) < temp_list_revealed_edges.index(spanning_tree_class.edge_l)) else spanning_tree_class.edge_h )
                         sampling_po = list(temp_list_revealed_edges)
                         sampling_po.remove(flag_edge)
                         
                         for edge in non_base_edges:
                             if spanning_tree[edge] == 1 and temp_list_revealed_edges.index(edge) < temp_list_revealed_edges.index(flag_edge):
                                 y_1.append(edge)
                                 sampling_po.remove(edge)
                             elif spanning_tree[edge] ==1 and temp_list_revealed_edges.index(edge) > temp_list_revealed_edges.index(flag_edge):
                                 y_2_selected.append(edge)
                                 if edge - (-1)**(edge % 2) in temp_list_unrevealed_edges:
                                     ranked_pairs_one.append((edge, edge - (-1)**(edge % 2)))
                                     assert(spanning_tree[edge - (-1)**(edge % 2)] == 0)
                             elif edge != spanning_tree_class.edge_l and edge != spanning_tree_class.edge_h:
                                 y_2_unselected.append(edge)
                         
                         for edge in temp_list_unrevealed_edges:
                             if spanning_tree[edge] == 1:
                                 Z.append(edge)
                                 remaining_edges.remove(edge)

                         for binary_sequence in it.product([0, 1], repeat=len(Z)):
                             #print("Binary Sequence:", binary_sequence)
                             ranked_pairs_two = []
                             z_1 = []
                             z_2 = []
                             two_prime = 0
                             for i in range (0,len(binary_sequence)):
                                 if binary_sequence[i] == 1:
                                     z_1.append(Z[i])
                                 else:
                                     z_2.append(Z[i])
                                     if Z[i] - (-1)**(Z[i] % 2) in list_revealed_edges:
                                         ranked_pairs_two.append((Z[i], Z[i] - (-1)**(Z[i] % 2)))
                                     else:
                                         two_prime += 1
                             
                             #sampling_po = list(y_2_selected)
                             #sampling_po.extend(y_2_unselected)
                             elements_to_sort = list(sampling_po)
                             elements_to_sort.extend(z_2)
                             elements_to_sort.extend(remaining_edges)
                             #print("Z is", Z,"elements_to_sort:", elements_to_sort, "partial order is:",sampling_po, "ranking 1,2:", ranked_pairs_one, ranked_pairs_two)
                             if len(z_1) == 0:
                                 easy_number = 1
                             else:
                                 easy_number =  math.factorial(len(y_1)+len(z_1))/math.factorial(len(y_1)) #* math.factorial(len(Z))/math.factorial(len(Z)-len(z_1)) 
                             crazy_number = LinearExtension(elements_to_sort, sampling_po, ranked_pairs_one, ranked_pairs_two)
                             spanning_tree_occurrences_single += easy_number*crazy_number/(2**(two_prime))
                             #print("easy and crazy are:",easy_number,crazy_number, "spanning tree occ:",spanning_tree_occurrences_single)
                             
                 spanning_tree_occurrences_all.append(spanning_tree_occurrences_single)
                    
            
        if 0 in list_revealed_edges:
            for spanning_tree in list_base_spanning_trees:
                #print("trees is :", spanning_tree)
                spanning_tree_occurrences_single = 0
                Z = []
                y_1 = []
                y_2_selected = []
                y_2_unselected = []
                ranked_pairs_one = []
                remaining_edges = list(list_unrevealed_edges)
                
                non_base_edges = list(list_revealed_edges)
                non_base_edges.remove(0)
                sampling_po = list(non_base_edges)
                
                for edge in non_base_edges:
                    if spanning_tree[edge] == 1 and list_revealed_edges.index(edge) < list_revealed_edges.index(0):
                        y_1.append(edge)
                        sampling_po.remove(edge)
                    elif spanning_tree[edge] ==1 and list_revealed_edges.index(edge) > list_revealed_edges.index(0):
                        y_2_selected.append(edge)
                        if edge - (-1)**(edge % 2) in list_unrevealed_edges:
                            ranked_pairs_one.append((edge, edge - (-1)**(edge % 2)))
                            assert(spanning_tree[edge - (-1)**(edge % 2)] == 0)
                    else:
                        y_2_unselected.append(edge)
                
                for edge in list_unrevealed_edges:
                    if spanning_tree[edge] == 1:
                        Z.append(edge)
                        remaining_edges.remove(edge)
                
                
                for binary_sequence in it.product([0, 1], repeat=len(Z)):
                    #print("Binary Sequence:", binary_sequence)
                    ranked_pairs_two = []
                    z_1 = []
                    z_2 = []
                    two_prime = 0
                    for i in range (0,len(binary_sequence)):
                        if binary_sequence[i] == 1:
                            z_1.append(Z[i])
                        else:
                            z_2.append(Z[i])
                            if Z[i] - (-1)**(Z[i] % 2) in list_revealed_edges:
                                ranked_pairs_two.append((Z[i], Z[i] - (-1)**(Z[i] % 2)))
                            else:
                                two_prime += 1
                    
                    #sampling_po = list(y_2_selected)
                    #sampling_po.extend(y_2_unselected)
                    elements_to_sort = list(sampling_po)
                    elements_to_sort.extend(z_2)
                    elements_to_sort.extend(remaining_edges)
                    #print("Z is", Z,"elements_to_sort:", elements_to_sort, "partial order is:",sampling_po, "ranking 1,2:", ranked_pairs_one, ranked_pairs_two)
                    if len(z_1) == 0:
                        easy_number = 1
                    else:
                        easy_number =  math.factorial(len(y_1)+len(z_1))/math.factorial(len(y_1)) #* math.factorial(len(Z))/math.factorial(len(Z)-len(z_1)) 
                    crazy_number = LinearExtension(elements_to_sort, sampling_po, ranked_pairs_one, ranked_pairs_two)
                    spanning_tree_occurrences_single += easy_number*crazy_number/(2**(two_prime))
                    #print("easy and crazy are:",easy_number,crazy_number, "spanning tree occ:",spanning_tree_occurrences_single)
                    
                spanning_tree_occurrences_all.append(spanning_tree_occurrences_single)
        else: 
            for spanning_tree in list_base_spanning_trees:
                #print("trees is :", spanning_tree)
                spanning_tree_occurrences_single = 0
                temp_index = 0
                for edge in list_revealed_edges:
                    if spanning_tree[edge] == 1:
                        assert(list_revealed_edges.index(edge) == temp_index)
                        temp_index += 1
                    else:
                        break
                for r in range (1,temp_index + 2):
                    temp_list_revealed_edges, temp_list_unrevealed_edges = self.AddEdgeToSample(0, r, list_revealed_edges, list_unrevealed_edges)
                    Z = []
                    y_1 = []
                    y_2_selected = []
                    y_2_unselected = []
                    ranked_pairs_one = []
                    remaining_edges = list(temp_list_unrevealed_edges)
                    
                    non_base_edges = list(temp_list_revealed_edges)
                    non_base_edges.remove(0)
                    sampling_po = list(non_base_edges)
                    
                    for edge in non_base_edges:
                        if spanning_tree[edge] == 1 and temp_list_revealed_edges.index(edge) < temp_list_revealed_edges.index(0):
                            y_1.append(edge)
                            sampling_po.remove(edge)
                        elif spanning_tree[edge] ==1 and temp_list_revealed_edges.index(edge) > temp_list_revealed_edges.index(0):
                            y_2_selected.append(edge)
                            if edge - (-1)**(edge % 2) in temp_list_unrevealed_edges:
                                ranked_pairs_one.append((edge, edge - (-1)**(edge % 2)))
                                assert(spanning_tree[edge - (-1)**(edge % 2)] == 0)
                        else:
                            y_2_unselected.append(edge)
                    
                    for edge in temp_list_unrevealed_edges:
                        if spanning_tree[edge] == 1:
                            Z.append(edge)
                            remaining_edges.remove(edge)
                    
                    for binary_sequence in it.product([0, 1], repeat=len(Z)):
                        #print("Binary Sequence:", binary_sequence)
                        ranked_pairs_two = []
                        z_1 = []
                        z_2 = []
                        two_prime = 0
                        for i in range (0,len(binary_sequence)):
                            if binary_sequence[i] == 1:
                                z_1.append(Z[i])
                            else:
                                z_2.append(Z[i])
                                if Z[i] - (-1)**(Z[i] % 2) in list_revealed_edges:
                                    ranked_pairs_two.append((Z[i], Z[i] - (-1)**(Z[i] % 2)))
                                else:
                                    two_prime += 1
                        
                        #sampling_po = list(y_2_selected)
                        #sampling_po.extend(y_2_unselected)
                        elements_to_sort = list(sampling_po)
                        elements_to_sort.extend(z_2)
                        elements_to_sort.extend(remaining_edges)
                        #print("Z is", Z,"elements_to_sort:", elements_to_sort, "partial order is:",sampling_po, "ranking 1,2:", ranked_pairs_one, ranked_pairs_two)
                        if len(z_1) == 0:
                            easy_number = 1
                        else:
                            easy_number =  math.factorial(len(y_1)+len(z_1))/math.factorial(len(y_1)) #* math.factorial(len(Z))/math.factorial(len(Z)-len(z_1)) 
                        crazy_number = LinearExtension(elements_to_sort, sampling_po, ranked_pairs_one, ranked_pairs_two)
                        spanning_tree_occurrences_single += easy_number*crazy_number/(2**(two_prime))
                        #print("easy and crazy are:",easy_number,crazy_number, "spanning tree occ:",spanning_tree_occurrences_single)
                        
                spanning_tree_occurrences_all.append(spanning_tree_occurrences_single)
                
        
        # Calculate Entropy
        occ_total = 0
        for occ in spanning_tree_occurrences_all:
            occ_total += occ
        
        H = 0
        for occ in spanning_tree_occurrences_all:
            H += occ/occ_total * np.log(occ_total/occ)
        return H
        #return spanning_tree_occurrences_all
                                
                    
                
        
        
        
        
    def AddEdgeToSample(self, edge, rank, list_revealed, list_unrevealed): #rank can be equal to 0???
        temp_list_picked_edges = list(list_revealed)
        temp_list_remaining_edges = list(list_unrevealed)   
        #print("revelead edges", temp_list_picked_edges, "unrevealed edges", temp_list_remaining_edges, "edge to add", edge, "rank", rank)
        temp_list_picked_edges.append(edge)
        temp_list_remaining_edges.remove(edge)
        temp_list = [edge]
        
        for i in range(rank, len(temp_list_picked_edges)):
            temp_list.append(temp_list_picked_edges[i-1])
        if rank >1:  
            temp_list_picked_edges = temp_list_picked_edges[:rank-1]
        else:
            temp_list_picked_edges = []
        temp_list_picked_edges.extend(temp_list)
        
        #print("NEW revelead edges", temp_list_picked_edges, "NEW unrevealed edges", temp_list_remaining_edges)
        
        return temp_list_picked_edges, temp_list_remaining_edges
        
    
    def Algorithm(self, list_revealed_edges, list_unrevealed_edges, has_edge, base_spanned = False, list_selected_edges = []):
        temp_list_revealed_edges = copy.deepcopy(list_revealed_edges)
        #print("revealed edges in algorithm=",temp_list_revealed_edges)
        temp_list_unrevealed_edges = copy.deepcopy(list_unrevealed_edges)   
        #print("unrevealed edges in algorithm=",temp_list_unrevealed_edges)
        if len(temp_list_revealed_edges) == self.edge_number: 
            return temp_list_revealed_edges, list_selected_edges
        else:
            entropy_dictionary = {}
            for edge in temp_list_unrevealed_edges:
                entropy = 0
                for rank in range(1, len(temp_list_revealed_edges) + 1): 
                    next_list_revealed_edges, next_list_unrevealed_edges = self.AddEdgeToSample(edge, rank, temp_list_revealed_edges, temp_list_unrevealed_edges)
                    entropy += self.CalculateEntropy(next_list_revealed_edges, next_list_unrevealed_edges)/len(temp_list_revealed_edges)    
                    del next_list_revealed_edges, next_list_unrevealed_edges
                entropy_dictionary[edge] = entropy
                
            next_edge = min(entropy_dictionary, key=lambda k: entropy_dictionary[k])
            if len(temp_list_revealed_edges) > 1:
                random_rank = random.randint(1, len(temp_list_revealed_edges))
            else:
                random_rank = 1 
                
            next_list_revealed_edges, next_list_unrevealed_edges = self.AddEdgeToSample(next_edge, random_rank, temp_list_revealed_edges, temp_list_unrevealed_edges)
            #print("next revealed edges in algorithm=",next_list_revealed_edges)
            #print("next unrevealed edges in algorithm=",next_list_unrevealed_edges)
 
            idx_hat = int((next_edge-1) / 2)
            if next_edge == 0 and not base_spanned:
                list_selected_edges.append(next_edge)
                base_spanned = True
            elif (not has_edge[idx_hat] or not base_spanned) and edge!=0:
                list_selected_edges.append(next_edge)
                base_spanned = has_edge[idx_hat] or base_spanned
                has_edge[idx_hat] = True
            # Recursive call of algorithm
            next_list_revealed_edges, list_selected_edges = self.Algorithm(next_list_revealed_edges, next_list_unrevealed_edges, has_edge, base_spanned, list_selected_edges)
            return next_list_revealed_edges, list_selected_edges