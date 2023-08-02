# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:34:53 2023

@author: matth
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx import create_empty_copy
import copy
import itertools as it


class EfficientFramework():
    def __init__(self, nodes):
        self.n = nodes
        self.g = nx.Graph()
        self.g.add_node(1, pos = (0,0))
        self.g.add_node(2, pos = (self.n*2,0))
        self.g.add_edge(1,2, index = 0)
        temp_iter = 2
        for i in range (3,self.n+1):
            self.g.add_node(i, pos = (self.n,i))
            self.g.add_edge(1,i, index = temp_iter-1)
            temp_iter += 1
            self.g.add_edge(2,i, index = temp_iter-1)
            temp_iter += 1
        self.m = temp_iter - 1
        self.edge_indices = np.arange(0, self.m)

    def DrawGraph(self, g, title):
        edge_labels = nx.get_edge_attributes(g, "weight")
        pos = nx.get_node_attributes(self.g,'pos')
        nx.draw_networkx_nodes(g, pos, node_size=10)
        nx.draw_networkx_edges(g, pos, g.edges())
        #nx.draw_networkx_labels(g, self.pos,font_size=8)
        nx.draw_networkx_edge_labels(g, pos, edge_labels,font_size=8)
        plt.title(title)
        
    # Function to Plot Graph    
    def PlotGraph(self, g, title = ''):
        self.DrawGraph(g, title)
        plt.show()
    
    # Function to Sample Edges
    def SampleEdges(self): # look at pre defined weights later on
        list_sampled_edges = []
        list_remaining_edges = list(self.edge_indices)

        sample_p = 0.5
        #random_seed = 42 
        rng_sample = np.random.default_rng()
        is_sampled = rng_sample.binomial(1, sample_p, self.m)
        
        #Pick edges from self.g
        for e in self.edge_indices:
            if is_sampled[e]:
                list_sampled_edges.append(e)
                list_remaining_edges.remove(e)
                
        return list_sampled_edges, list_remaining_edges
    
    # Function to Calculate Maximum Weight Spannint Tree
    def CalculateMST(self, ordering):
        OPT = []
        decimal_encoding_OPT = 0
        has_edge = [False for i in range(self.n-1)]
        base_spanned = False
        for edge in ordering:
            idx_hat = int((edge-1) / 2)
            if edge == 0 and not base_spanned:
                OPT.append(edge)
                decimal_encoding_OPT += 2**(edge)
                base_spanned = True
            elif (not has_edge[idx_hat] or not base_spanned) and edge!=0:
                OPT.append(edge)
                decimal_encoding_OPT += 2**(edge)
                base_spanned = has_edge[idx_hat] or base_spanned
                has_edge[idx_hat] = True

        return OPT, decimal_encoding_OPT
    
    # Function to Calculate (Conditional) Entropy
    def CalculateEntropy(self, list_picked_edges, list_remaining_edges):
        temp_list_picked_edges = list(list_picked_edges)
        temp_list_remaining_edges = list(list_remaining_edges)    
        spanning_tree_dictionary = {}
        total_number_rankings = 0
        non_sample_size = len(temp_list_remaining_edges)
        
        # Go over all Linear Extensions of the Partial Order Induced by Revealed Edges
        # Calculate OPT for every such Linear Extension
        # Count Occurrences of every Possible Realisation of OPT
        for non_sample_w in it.permutations(temp_list_remaining_edges):
            for interleaving in it.combinations(range(0, self.m), len(non_sample_w)):
                total_number_rankings +=1
                all_w = [0 for i in range(self.m)]
                idx_interleaving, idx_sample, idx_non_sample = 0, 0, 0
                for i in range(self.m):
                    if idx_interleaving < non_sample_size and i == interleaving[idx_interleaving]:
                        all_w[i] = non_sample_w[idx_non_sample]
                        idx_interleaving += 1
                        idx_non_sample += 1
                    else:
                        assert(0 <= idx_sample < len(temp_list_picked_edges))
                        all_w[i] = temp_list_picked_edges[idx_sample]
                        idx_sample += 1

                temp_OPT, temp_decimal_encoding_OPT = self.CalculateMST(all_w)
                spanning_tree_dictionary[temp_decimal_encoding_OPT] = spanning_tree_dictionary.get(temp_decimal_encoding_OPT, 0) + 1
        
        # Calculate the Conditional Entropy
        probabilities = []
        values = 0
        for value in spanning_tree_dictionary.values():
            probabilities.append(value/total_number_rankings)
            values += value
        H = 0
        for i in range(0, len(probabilities)):
            H += probabilities[i]*np.log(1/probabilities[i])
            
        return H 
        
    # Function to Reveal an Edge
    def AddEdgeToSample(self, edge, rank, list_revealed, list_unrevealed):
        assert(edge in list_unrevealed and edge not in list_revealed)
        temp_list_picked_edges = list(copy.deepcopy(list_revealed))
        temp_list_remaining_edges = list(copy.deepcopy(list_unrevealed))    
        temp_list_remaining_edges.remove(edge)
        temp_list = [edge]
        
        for i in range(rank, len(temp_list_picked_edges) + 1):
            temp_list.append(temp_list_picked_edges[i - 1])
        if rank > 1:  
            temp_list_picked_edges = temp_list_picked_edges[:rank - 1]
        else:
            temp_list_picked_edges = []
        temp_list_picked_edges.extend(temp_list)
                
        return temp_list_picked_edges, temp_list_remaining_edges
        
    # Implementation of the MaximumInformationRevealing Algorithm
    def Algorithm(self, total_ranks, list_revealed_edges, list_unrevealed_edges, has_edge, base_spanned = False, list_selected_edges = []):
        temp_list_revealed_edges = list(list_revealed_edges)
        temp_list_unrevealed_edges = list(list_unrevealed_edges)   
        temp_list_selected_edges = list(list_selected_edges)
        
        # Once all Edges Revealed, Return
        if len(temp_list_revealed_edges) == self.m: 
            return temp_list_revealed_edges, temp_list_selected_edges
        
        else:
            # Find Edge of Maximal Expected Information Content --> Next Edge
            entropy_dictionary = {}
            for edge in temp_list_unrevealed_edges:
                entropy = 0
                for rank in range(1, len(temp_list_revealed_edges) + 2): 
                    next_list_revealed_edges, next_list_unrevealed_edges = self.AddEdgeToSample(edge, rank, temp_list_revealed_edges, temp_list_unrevealed_edges)
                    entropy += self.CalculateEntropy(next_list_revealed_edges, next_list_unrevealed_edges)/len(next_list_revealed_edges)    
                    del next_list_revealed_edges, next_list_unrevealed_edges
                entropy_dictionary[edge] = entropy
            next_edge = min(entropy_dictionary, key=lambda k: entropy_dictionary[k])
            
            # Find Rank of Next Edge in Set of Revealed Edges
            rank_next_edge = 1
            for i in range (0,len(total_ranks)):
                if total_ranks[i] in temp_list_revealed_edges:
                    rank_next_edge += 1
                if total_ranks[i] == next_edge:
                    break
            
            # Reveal Next Edge 
            next_list_revealed_edges, next_list_unrevealed_edges = self.AddEdgeToSample(next_edge, rank_next_edge, temp_list_revealed_edges, temp_list_unrevealed_edges)
            del temp_list_revealed_edges, temp_list_unrevealed_edges
            
            # Include Newly Revealed Edge if in Current OPT and Does Not Close Cycle
            OPT_current, OPT_current_encoding = self.CalculateMST(next_list_revealed_edges)
            idx_hat = int((next_edge - 1) / 2)
            if next_edge == 0 and not base_spanned and next_edge in OPT_current:
                temp_list_selected_edges.append(next_edge)
                base_spanned = True
            elif (not has_edge[idx_hat] or not base_spanned) and edge!=0 and next_edge in OPT_current:
                temp_list_selected_edges.append(next_edge)
                base_spanned = has_edge[idx_hat] or base_spanned
                has_edge[idx_hat] = True
            
            # Recurse
            next_list_revealed_edges, temp_list_selected_edges_outp = self.Algorithm(total_ranks, next_list_revealed_edges, next_list_unrevealed_edges, has_edge, base_spanned, temp_list_selected_edges)
            del temp_list_selected_edges
            
            # Return Revealed and Selected Edges 
            return next_list_revealed_edges, temp_list_selected_edges_outp
        

