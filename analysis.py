#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
my_site = os.path.join(os.environ["HOME"], ".local/lib/python2.7/site-packages")
sys.path.insert(0, my_site)

import h5py
import networkx as nx
import numpy as np
import pycuda.driver as cuda
import scipy.stats as st
import sys

import aux
from consts import *

def to_graph(connections):
    graph = nx.DiGraph()
    
    ca_size = connections.shape[0]
    
    for cell in xrange(ca_size):
        for neighbor in connections[cell]:
            graph.add_edge(neighbor, cell)
        
        # Count the number of rewired connection this cell has
        graph.node[cell]['rew'] = (connections[cell] !=
            (np.arange(cell - 3, cell + 4) % ca_size)).sum()

    return graph


class AnalysisIndividual:
    
    __cuda_module = False
    
    def __init__(self, individual, correct, executions, ca_size,
                 connection_radius, ca_iterations, ca_repeat, k_history,
                 save_executions=0):
        
        self.__ca_size = ca_size
        self.__connection_radius = connection_radius
        self.__n_connections = 2 * self.__connection_radius + 1
        self.__ca_iterations = ca_iterations
        self.__ca_repeat = ca_repeat
        self.__k_history = k_history
        self.__n_possible_history = 2 ** self.__k_history
        self.__n_observations = self.__ca_repeat * \
            (self.__ca_iterations - self.__k_history + 1)
        self.__save_executions = save_executions
        
        self.__individual = individual
        self.__individual_number = self.__individual.number
        self.__rules = self.__individual.gene_rules
        self.__connections = self.__individual.connections
        self.__graph = to_graph(self.__connections)
        self.__executions = executions
        
        density = np.mean(self.__executions[:, 0], axis=1)
        self.__majority = np.round(density).astype(np.uint32)
        # The closer the density is to .5 the harder the configuration is to
        # decide
        self.__difficult = 1 - np.abs(density - .5) / .5
        # Checking which repetitions converged to a single state
        self.__converged = np.all(self.__executions[:, -1] == 
                                  self.__executions[:, -1, 0].reshape(-1, 1),
                                  axis=1)
        # Checking how many cells in each repetition converged to the right
        # state
        self.__cells_correct = np.mean(self.__executions[:, -1] ==
                                       self.__majority.reshape(-1, 1), axis=1)
        self.__correct = correct
        self.__fitness = np.mean(self.__correct)
        
        self.__gini = None
        self.__limits = None
        self.__entropy_rate = None
        self.__base_table = None
        self.__correlations = None
        
        # Initialize the CUDA module
        if not AnalysisIndividual.__cuda_module:
            AnalysisIndividual.__cuda_module = True
            
            cuda_module = aux.CudaModule('analysis.cu',
                                         (self.__ca_size, self.__ca_iterations,
                                          self.__ca_repeat,
                                          self.__connection_radius,
                                          self.__n_connections,
                                          self.__n_observations,
                                          self.__k_history,
                                          self.__n_possible_history))
            AnalysisIndividual.__kernel_calc_diffs = \
                cuda_module.get_function("kernel_calc_diffs")
            AnalysisIndividual.__kernel_probabilities = \
                cuda_module.get_function("kernel_probabilities")
            AnalysisIndividual.__kernel_active_storage = \
                cuda_module.get_function("kernel_active_storage")
            AnalysisIndividual.__kernel_entropy_rate = \
                cuda_module.get_function("kernel_entropy_rate")
    
    def __calculate_gini(self, values):
        # Calculate the Gini coefficient to measure the inequality in a
        # distribution of values
        cum_values = np.sort(values).cumsum()
        
        return 1 - (cum_values[0] + (cum_values[1:] + cum_values[:-1]).sum()) \
            / float(cum_values[-1] * cum_values.size)
    
    def __get_limits(self):
        # This function implements a heuristic to calculate how many times a
        # cell has the role of "limit" of a diffusion in a simulation.
        # The main idea here is that, usually, information in cellular automata
        # flows in a given direction at a constant speed. If we know this
        # direction and speed, we can check how many times a cell interrupts a
        # flow.
        sum_diffs = np.zeros(self.__ca_size, dtype=np.uint32)
        
        try:
            self.__kernel_calc_diffs(cuda.In(self.__majority),
                                     cuda.In(self.__executions),
                                     cuda.InOut(sum_diffs),
                                     block=(self.__ca_size, 1, 1), grid=(1,))
            cuda.Context.synchronize()
        except cuda.Error as e:
            sys.exit("CUDA: Execution failed ('%s')!" % e)
        
        # For all repetitions, calculate the ratio of total iterations each
        # cell acted as a "limit"
        self.__limits = sum_diffs / \
            float(self.__ca_repeat * self.__ca_iterations)
    
    def get_individual_info(self):
        if self.__gini != None:
            # If all metrics are already computed, just return them!
            return self.__fitness, self.__gini, self.__prop_max_min, \
                self.__individual.epoch, self.__individual_number, \
                self.__clustering, self.__average_k_neigh, \
                self.__average_shortest_path, self.__diameter
        
        self.__get_limits()
        self.__gini = self.__calculate_gini(self.__limits)
        self.__prop_max_min = self.__limits.max() / self.__limits.min()
        
        # As clustering coefficient is not defined for directed graphs, we
        # convert the graph to its undirected version
        self.__clustering = nx.average_clustering(nx.Graph(self.__graph))
        self.__average_shortest_path = \
            nx.average_shortest_path_length(self.__graph)
        
        try:
            self.__diameter = nx.diameter(self.__graph)
        except nx.exception.NetworkXError:
            self.__diameter = float('nan')
        
        self.__convergence = np.mean(self.__converged)
        
        table_individual = {
            # Serial number
            "i_num": np.array([self.__individual_number], dtype=np.int),
            # Individual fitness
            "fit": np.array([self.__fitness], dtype=np.float),
            # Ratio of the repetitions that converged to a single state
            "conv": np.array([self.__convergence], dtype=np.float),
            # gini and max_min are metrics intended to measure the inequality
            # in the number of times each cell is a "limit"
            "gini": np.array([self.__gini], dtype=np.float),
            "max_min": np.array([self.__prop_max_min], dtype=np.float),
            # Epoch in the evolution
            "epoch": np.array([self.__individual.epoch], dtype=np.float),
            # Clustering coefficient
            "clust": np.array([self.__clustering], dtype=np.float),
            # Average shortests path between each pair of cells
            "short": np.array([self.__average_shortest_path], dtype=np.float),
            # Maximum distance between any two cells
            "diam": np.array([self.__diameter], dtype=np.float)}
        
        return table_individual
    
    def __get_probs_entropy(self):
        # Calculate information theoretical metrics to evaluate the
        # computational role of each cell
        if self.__entropy_rate != None:
            # If all metrics are already computed, just return them!
            return self.__entropy_rate, self.__active_storage, \
                self.__cond_entropy
        
        p_joint_table = np.zeros((self.__ca_size, self.__n_possible_history,
                                  2), dtype=np.float32)
        p_prev_table = np.zeros((self.__ca_size, self.__n_possible_history), 
                                dtype=np.float32)
        p_curr_table = np.zeros((self.__ca_size, 2), dtype=np.float32)
        
        try:
            self.__kernel_probabilities(cuda.In(self.__executions), 
                                        cuda.InOut(p_joint_table),
                                        cuda.InOut(p_prev_table),
                                        cuda.InOut(p_curr_table),
                                        block=(self.__ca_size, 1, 1),
                                        grid=(self.__ca_repeat, 1, 1))
            cuda.Context.synchronize()
        except cuda.Error as e:
            sys.exit("CUDA: Execution failed!\n'%s'" % e)
        
        # The entropy rate is a measure of the uncertainty in a cell's state
        # given its past
        self.__entropy_rate = np.zeros(self.__ca_size, dtype=np.float32)
        # The active information storage is the amount of past information
        # currently in use by a cell, i.e., its memory
        self.__active_storage = np.zeros(self.__ca_size, dtype=np.float32)
    
        try:
            self.__kernel_entropy_rate(cuda.In(p_joint_table),
                                       cuda.In(p_prev_table),
                                       cuda.InOut(self.__entropy_rate),
                                       block=(self.__ca_size, 1, 1))
            cuda.Context.synchronize()
        
            for i in xrange(self.__ca_iterations - self.__k_history):
                ca_aux = np.array(self.__executions[:,
                                                    i:i + self.__k_history + 1,
                                                    :])
                self.__kernel_active_storage(cuda.In(ca_aux),
                                             cuda.In(p_joint_table),
                                             cuda.In(p_prev_table),
                                             cuda.In(p_curr_table),
                                             cuda.InOut(self.__active_storage),
                                             block=(self.__ca_size, 1, 1),
                                             grid=(self.__ca_repeat, 1, 1))
                cuda.Context.synchronize()
        except cuda.Error as e:
            sys.exit("CUDA: Execution failed!\n'%s'" % e)
        
        aux = np.multiply(p_joint_table, np.log2(np.divide(p_prev_table.
                          reshape(p_prev_table.shape + (1,)), p_joint_table)))
        aux[p_joint_table == 0] = 0
        self.__cond_entropy = np.sum(aux, axis=(1, 2)) / self.__n_observations

        return self.__entropy_rate, self.__active_storage, self.__cond_entropy

    def get_cells_info(self):
        self.__get_limits()
        self.__get_probs_entropy()
        
        full_data = {
            "lim": self.__limits,
            "ent_rt": self.__entropy_rate,
            "act_st": self.__active_storage,
            "cond_ent": self.__cond_entropy}
        
        if self.__base_table == None:
            # Calculate graph measures
            order = sorted(self.__graph.nodes())
            
            pagerank = nx.pagerank(self.__graph)
            pagerank = np.array([pagerank[k] for k in order], dtype=np.float)
            
            try:
                hubs, authorities = nx.hits(self.__graph, 1000)
                hubs = np.array([hubs[k] for k in order], dtype=np.float)
                authorities = np.array([authorities[k] for k in order],
                                       dtype=np.float)
            except nx.exception.NetworkXError:
                hubs = np.repeat(float('nan'), self.__ca_size).astype(np.float)
                authorities = hubs
            
            try:
                eccentricity = nx.eccentricity(self.__graph)
                eccentricity = np.array([eccentricity[k] for k in order], 
                                        dtype=np.float)
            except nx.exception.NetworkXError:
                eccentricity = np.repeat(float('nan'), self.__ca_size). \
                    astype(np.float)
            
            closeness = nx.closeness_centrality(self.__graph)
            closeness = np.array([closeness[k] for k in order], dtype=np.float)
            closeness_reverse = nx.closeness_centrality(
                self.__graph.reverse(True))
            closeness_reverse = np.array([closeness_reverse[k] for k in order],
                                         dtype=np.float)
            betweenness = nx.betweenness_centrality(self.__graph)
            betweenness = np.array([betweenness[k] for k in order], 
                                   dtype=np.float)
            
            try:
                eigenvector = nx.eigenvector_centrality(self.__graph, 1000)
                eigenvector = np.array([eigenvector[k] for k in order], 
                                       dtype=np.float)
            except nx.exception.NetworkXError:
                eigenvector = np.repeat(float('nan'), self.__ca_size). \
                    astype(np.float)
            
            load = nx.load_centrality(self.__graph)
            load = np.array([load[k] for k in order], dtype=np.float)
            clustering = nx.clustering(nx.Graph(self.__graph))
            clustering = np.array([clustering[k] for k in order], 
                                  dtype=np.float)
            in_degree = nx.in_degree_centrality(self.__graph)
            in_degree = np.array([in_degree[k] for k in order], dtype=np.float)
            out_degree = nx.out_degree_centrality(self.__graph)
            out_degree = np.array([out_degree[k] for k in order], 
                                  dtype=np.float)
            rewires = np.array([self.__graph.node[k]['rew'] for k in order], 
                               dtype=np.float)
            average_k_neigh = nx.average_neighbor_degree(self.__graph)
            average_k_neigh = np.array([average_k_neigh[k] for k in order], 
                                       dtype=np.float)
            
            self.__base_table = {
                "epoch": np.repeat(self.__individual.epoch, self.__ca_size). \
                    astype(np.int),
                "i_num": np.repeat(self.__individual_number, self.__ca_size). \
                    astype(np.int),
                "pr": pagerank,
                "hub": hubs,
                "auth": authorities,
                "ecc": eccentricity,
                "cls": closeness,
                "cls_rev": closeness_reverse,
                "btw": betweenness,
                "eig": eigenvector,
                "load": load,
                "cltr": clustering,
                "ind": in_degree,
                "outd": out_degree,
                "rew": rewires,
                "kneigh": average_k_neigh}
        
        return dict(full_data.items() + self.__base_table.items())
    
    def save_executions(self):
        # Save space-time diagrams of some executions
        for i in np.random.choice(range(self.__executions.shape[0]),
                                  self.__save_executions, replace=False):
            
            aux.save_as_image(self.__executions[i],
                              "images/i%04d" % self.__individual_number,
                              "execution-%06d.png" % i)

class Analysis:
    
    elems = 0
    
    def __init__(self, data_file, ca_size, ca_iterations, ca_repeat, 
                 connection_radius, k_history, save_executions=0):
        
        self.__ca_size = ca_size
        self.__ca_iterations = ca_iterations
        self.__ca_repeat = ca_repeat
        self.__connection_radius = connection_radius
        self.__k_history = k_history
        self.__save_executions = save_executions
        self.__data_file = h5py.File(data_file, "w-")
    
    def add_individual(self, individual):
        # Run simulations with densities uniformly distributed in [0, 1],
        # storing execution data for posterio analysis
        correct, executions = individual.get_execution_data(UNIFORM_RHO)
        
        # Perform individual analysis
        individual = AnalysisIndividual(individual, correct, executions,
                                        self.__ca_size,
                                        self.__connection_radius,
                                        self.__ca_iterations, self.__ca_repeat,
                                        self.__k_history,
                                        save_executions=self.__save_executions)
        
        Analysis.elems += 1
        
        table_cells = individual.get_cells_info()
        table_individual = individual.get_individual_info()
        
        individual.save_executions()
        
        del correct
        del executions
        del individual
        
        # Store the individual analysis in a HDF5 file
        group = self.__data_file.create_group("individual%d" %
                                              table_individual["i_num"])
        cells_grp = group.create_group("cells")
        
        for key, values in table_cells.iteritems():
            cells_grp.create_dataset(key, data=values, shape=values.shape,
                                     dtype=values.dtype)
        
        individuals_grp = group.create_group("individuals")
        
        for key, values in table_individual.iteritems():
            individuals_grp.create_dataset(key, data=values, 
                                           shape=values.shape,
                                           dtype=values.dtype)
        
        self.__data_file.flush()
    
    def get_table(self):
        table = {
            "cells": {},
            "individuals": {}}
        
        for individual_grp in self.__data_file.values():
            for group in ["cells", "individuals"]:
                for key, values in individual_grp[group].iteritems():
                    try:
                        table[group][key].append(values.value)
                    except KeyError:
                        table[group][key] = [values.value]
        
        for group_values in table.values():
            for key, values in group_values.iteritems():
                group_values[key] = np.concatenate(values)
        
        return table
    
    def get_correlations(self):
        table = self.get_table()
        
        correlations = {'cells': {}, 'individuals': {}}
        
        refs_cells = ['lim', 'cls_rev']
        
        for ref in refs_cells:
            correlations['cells'][ref] = {}
            ref_cell = table['cells'][ref]
        
            for key, values in table['cells'].iteritems():
                if key == ref:
                    continue
                
                correlations['cells'][ref][key] = \
                    st.spearmanr(ref_cell, values)
        
        refs_individuals = ['gini', 'max_min', 'short', 'fit']
        
        for ref in refs_individuals:
            correlations['individuals'][ref] = {}
            ref_individual = table['individuals'][ref]
            
            for key, values in table['individuals'].iteritems():
                if key == ref:
                    continue
                
                correlations['individuals'][ref][key] = \
                    st.spearmanr(ref_individual, values)
        
        return correlations
