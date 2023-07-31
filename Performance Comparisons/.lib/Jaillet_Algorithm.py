# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 00:53:17 2023

@author: matth
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx import create_empty_copy
import copy


class Framework():
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
    def Algorithm(self, total_ranks, list_revealed_edges, list_unrevealed_edges):
        temp_list_revealed_edges = list(list_revealed_edges)
        temp_list_unrevealed_edges = list(list_unrevealed_edges)   
        
        index = 0
        list_selected_edges = []
        list_A = []
        base_spanned_sample = False
        base_spanned_selection = False
        has_edge_sample = np.zeros(self.m - 1, dtype=bool)
        has_edge_selection = np.zeros(self.m - 1, dtype=bool)
        
        for a in temp_list_revealed_edges:
            index = temp_list_revealed_edges.index(a)
            if a == 0:
                base_spanned_sample = True
                break
            elif has_edge_sample[int((a - 1) / 2)] == True:
                base_spanned_sample = True
                break
            else: 
                has_edge_sample[int((a - 1) / 2)] = True
        
        # Finding the elements of first non-empty span(A_i)
        temp_A = []
        if index > 0 and base_spanned_sample == True:
            if 0 in temp_list_unrevealed_edges:
                temp_A.append(0)
            for i in range(0, index):
                a = temp_list_revealed_edges[i]
                if a - (-1)**(a % 2) in temp_list_unrevealed_edges and a != 0:
                    temp_A.append(a - (-1)**(a % 2))
            if len(temp_A) > 0:
                temp_A.append(temp_list_revealed_edges[index])
                list_A.append(temp_A)
        #print(base_spanned_sample)
            
        # Finding the elements of Remaining span(A_i)
        for i in range(index + 1, len(temp_list_revealed_edges)):
            temp_A = []
            a = temp_list_revealed_edges[i]
            if a - (-1)**(a % 2) in temp_list_unrevealed_edges:
                temp_A.append(a - (-1)**(a % 2))
                temp_A.append(a)
                list_A.append(temp_A)
                
        #print(list_A)
        # First Phase of Algorithm
        for A in list_A:
            for a in A[:-1]:
                temp_list_unrevealed_edges.remove(a)
                if has_edge_selection[int((a - 1) / 2)] == False or base_spanned_selection == False:
                    if total_ranks.index(a) < total_ranks.index(A[-1]):
                        list_selected_edges.append(a)
                        base_spanned_selection = has_edge_selection[int((a - 1) / 2)]
                        has_edge_selection[int((a - 1) / 2)] = True
                        
        # Second Phase of Algorithm
        for i in range(0, len(temp_list_unrevealed_edges)):
            a = temp_list_unrevealed_edges[0]
            temp_list_unrevealed_edges.remove(a)
            if has_edge_selection[int((a - 1) / 2)] == False or base_spanned_selection == False:
                list_selected_edges.append(a)
                base_spanned_selection = has_edge_selection[int((a - 1) / 2)]
                has_edge_selection[int((a - 1) / 2)] = True
            
        #print("CALCULATED SELECTION is:", list_selected_edges)
        return list_selected_edges
    
