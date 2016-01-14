#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pycuda.driver as cuda
import random
import sys

import aux
from consts import *

class Individual:
    
    __number = 0
    
    def __init__(self, ca_size, connection_radius, n_connections, 
                 ca_iterations, ca_repeat, mutation_rate, standard_deviation,
                 epoch, kernel_simulation, possible_connections, *args,
                 **kwargs):
        
        n_args = len(kwargs)
        
        # Individual's serial number
        self.number = Individual.__number
        Individual.__number += 1
        
        self.__ca_size = ca_size
        self.__connection_radius = connection_radius
        self.__n_connections = n_connections
        self.__size_rules = 2 ** self.__n_connections
        self.__ca_iterations = ca_iterations
        self.__ca_repeat = ca_repeat
        self.epoch = epoch
        self.__kernel_simulation = kernel_simulation
        self.__possible_connections = possible_connections
        
        if n_args == 1:
            # If the individual has no parents then we create a random
            # individual with the specified amount of rewiring
            p = kwargs['p']
            
            # Create a random rule
            gene_rules = np.random.randint(2, size=self.__size_rules). \
                astype(np.uint32)
            
            n_rwr = int(self.__ca_size * self.__n_connections * p)
            gene_rewires = []
            
            # Choose n_rewires connections to rewire
            for rewire in random.sample(self.__possible_connections, n_rwr):
                # Rewire the connection to a cell randomly chosen between all
                # existing cells - notice that a cell can have multiple
                # connections to the same cell as well as self-loops, however,
                # a connection can be rewired once
                gene_rewires.append(rewire + 
                                    (np.random.randint(self.__ca_size),))
        elif n_args == 2:
            # If the individual has two parents then we create an individual
            # using crossover and mutation
            father = kwargs['father']
            mother = kwargs['mother']
            
            # The number of rewirings is a convex combination of the number
            # of rewirings in the father and in the mother. In our experiments,
            # both the parents have the same number of rewirings, so the
            # following lines are quite redundant; however, the code is ready
            # for experiments where we allow individuals with a different
            # number of rewirings.
            parenthood = np.random.rand()
            p = parenthood * father.p + (1 - parenthood) * mother.p + \
                np.random.randn() * standard_deviation
            p = min(1., max(0., p))
            n_rewires = int(self.__ca_size * self.__n_connections * p)
            
            # Select each rule from one parent with equal probability
            gene_rules = np.empty(father.gene_rules.shape, dtype=np.uint32)
            from_father = (np.random.rand(*gene_rules.shape) < .5)
            gene_rules[from_father] = father.gene_rules[from_father]
            gene_rules[- from_father] = mother.gene_rules[- from_father]
            # Flip some rules' bits to mutate them
            mutate = (np.random.rand(*gene_rules.shape) < 1. / mutation_rate)
            gene_rules[mutate] = 1 - gene_rules[mutate]
            
            # Select the rewirings from the rewires from both the parents
            possible_rewires = father.gene_rewires + mother.gene_rewires
            possible_mutations = self.__possible_connections
            gene_rewires = []
            
            for i in xrange(n_rewires):
                if np.random.random() < 1. / mutation_rate:
                    # If this rewire is going to be mutated, then we rewire
                    # a random connection to a random cell
                    new = random.choice(possible_mutations) + \
                        (np.random.randint(self.__ca_size),)
                else:
                    # Select a rewire to be copied from one of the parents with
                    # uniform probability
                    new = random.choice(possible_rewires)
                
                gene_rewires.append(new)
                
                # Avoid to rewire any connection twice, be it through crossover
                # or mutation
                possible_rewires = [pr for pr in possible_rewires \
                    if pr[0:2] != new[0:2]]
                possible_mutations = [pm for pm in possible_mutations \
                    if pm != new[0:2]]
        else:
            raise Exception("Individual contructor needs sufficient arguments "
                            "about the individual to be created")
        
        self.gene_rules = gene_rules
        self.gene_rewires = sorted(gene_rewires)
        self.p = p
        
        # Create an adjacency list for a ring graph
        self.connections = ((np.arange(self.__ca_size).reshape((-1, 1)) + 
            (np.arange(self.__n_connections).reshape((1, -1)) -
            self.__connection_radius)) % self.__ca_size).astype(np.uint32)
        
        # Apply the rewires to the adjacency list
        for rewire in self.gene_rewires:
            self.connections[rewire[0], rewire[1]] = rewire[2]
        
        # Evaluate individual's fitness in a scenarion in which the initial
        # density is uniformly sampled from [0, 1]
        self.fitness = self.evaluate(ic_type=UNIFORM_RHO)
    
    def __evaluate(self, ic_type, analysis):
        # We send a random array to the GPU, as it has no access to a
        # pseudo-random number generator as we have in the CPU
        random_arr = np.random.rand(self.__ca_repeat, self.__ca_size). \
            astype(np.float32)
        # Array where the code running in the GPU will write 1 in i-th position
        # if the i-th repetition converged to the correct answer
        correct = np.zeros(self.__ca_repeat, dtype=np.int32)
        
        if analysis:
            # If this execution is going to be performed for posterior analysis
            # we store all the iterations of each repetition of the cellular
            # automaton
            executions = np.zeros((self.__ca_repeat, self.__ca_iterations + 1,
                                   self.__ca_size), dtype=np.uint32)
        else:
            executions = np.zeros(1)
        
        try:
            # Let's simulate on the GPU!
            self.__kernel_simulation(cuda.In(self.gene_rules),
                                     cuda.In(self.connections),
                                     cuda.In(random_arr),
                                     cuda.InOut(correct), np.int32(self.epoch),
                                     np.int32(ic_type), np.int32(analysis),
                                     cuda.Out(executions),
                                     block=(self.__ca_size, 1, 1),
                                     grid=(self.__ca_repeat, 1))
            cuda.Context.synchronize()
        except cuda.Error as e:
            sys.exit("CUDA: Execution failed!\n'%s'" % e)
    
        fitness = correct.mean()
        
        if analysis:
            return correct, executions
        
        return fitness
    
    def evaluate(self, ic_type=UNIFORM_RHO):
        return self.__evaluate(ic_type=ic_type, analysis=False)
    
    def get_execution_data(self, ic_type=CONSTANT_RHO):
        return self.__evaluate(ic_type=ic_type, analysis=True)

class Evolution:
    
    def __init__(self, p_rewire, population_size, max_epochs, tourney_size, 
                 mutation_rate, standard_deviation, ca_size, connection_radius, 
                 ca_iterations, ca_repeat):
        
        self.__p_rewire = p_rewire
        self.__population_size = population_size
        self.__max_epochs = max_epochs
        self.__tourney_size = tourney_size
        self.__mutation_rate = mutation_rate
        self.__standard_deviation = standard_deviation
        self.epoch = 0
        
        self.__ca_size = ca_size
        self.__connection_radius = connection_radius
        self.__n_connections = 2 * self.__connection_radius + 1
        self.__ca_iterations = ca_iterations
        self.__ca_repeat = ca_repeat
        
        # Each cell have connections to its connection_radius neighbors to the
        # left and to the right (i.e., a total of n_connections)
        self.__possible_connections = list(itertools.product(
            range(self.__ca_size), range(self.__connection_radius) +
            range(self.__connection_radius + 1, self.__n_connections)))
        
        # Load the function to simulate the execution of a CA in the GPU
        cuda_module = aux.CudaModule('cellular_automata.cu',
                                     (self.__ca_size, self.__ca_iterations,
                                      self.__ca_repeat, self.__n_connections,
                                      self.__max_epochs, CONSTANT_RHO,
                                      UNIFORM_RHO, DECREASING_RHO))
        self.__kernel_simulation = \
            cuda_module.get_function("kernel_simulation")
        
        self.__individuals = [Individual(self.__ca_size, 
                                         self.__connection_radius,
                                         self.__n_connections,
                                         self.__ca_iterations,
                                         self.__ca_repeat,
                                         self.__mutation_rate, 
                                         self.__standard_deviation, self.epoch,
                                         self.__kernel_simulation, 
                                         self.__possible_connections, 
                                         p=self.__p_rewire)
                              for i in xrange(self.__population_size)]
        self.__individuals.sort(key=lambda x: x.fitness)
    
    def __tourney(self):
        # The list of individuals is sorted by fitness!
        return np.random.randint(self.__population_size, 
                                 size=self.__tourney_size).max()
    
    def __reproduction(self):
        father = self.__individuals[self.__tourney()]
        mother = self.__individuals[self.__tourney()]
    
        return Individual(self.__ca_size, self.__connection_radius,
                          self.__n_connections, self.__ca_iterations, 
                          self.__ca_repeat, self.__mutation_rate, 
                          self.__standard_deviation, self.epoch, 
                          self.__kernel_simulation, 
                          self.__possible_connections, father=father, 
                          mother=mother)
    
    def finished(self):
        return self.epoch == self.__max_epochs
    
    def run(self, iterations=1):
        for self.epoch in xrange(self.epoch + 1, self.epoch + iterations + 1):
            self.__individuals = [self.__reproduction()
                                  for i in xrange(self.__population_size)]
            self.__individuals.sort(key=lambda x: x.fitness)
    
    def get_individuals(self):
        return self.__individuals
