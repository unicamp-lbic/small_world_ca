#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import numpy as np
import sys
import time

import analysis
import aux
from consts import *
import evolution

def log_iter(p_rewire, individuals):
    best_individual = individuals[-1]
    
    # Evaluate the best individual using densities around 0.5 (the most
    # difficult situation in the density classification task)
    best_fitness_rho05 = best_individual.evaluate(ic_type=CONSTANT_RHO)
    all_fitnesses = [i.fitness for i in individuals]
    
    logging.info("Epoch %d: best=%.2f / best(rho=0.5)=%.2f / avg=%.2f / "
                 "median=%.2f / worst=%.2f / time=%.2fmin" %
                 (individuals[0].epoch, best_individual.fitness, 
                  best_fitness_rho05, np.mean(all_fitnesses), 
                  np.median(all_fitnesses), individuals[0].fitness, 
                  time.clock() / 60.))
    
    # Print the rules and rewired edges for the best individual
    logging.info("Best genotype: {%s, %s, %s}" % (best_individual.number,
                 "".join(map(str, best_individual.gene_rules)), 
                 best_individual.gene_rewires))

def analysis_p(p_rewire, population_size, max_epochs, tourney_size,
               mutation_rate, standard_deviation, ca_size, connection_radius,
               ca_iterations, ca_repeat, k_history, save_executions):
    
    logging.info("Simulation for rewiring=%.3f%%" % (100 * p_rewire))
    
    data_file = aux.get_root_folder() + '/data.p%.3f.hdf5' % p_rewire
    logging.info("Saving evolution data in file '%s'" % data_file)
    
    analytics = analysis.Analysis(data_file, ca_size, ca_iterations, ca_repeat,
                                  connection_radius, k_history, 
                                  save_executions)
    
    evolve = evolution.Evolution(p_rewire, population_size, max_epochs,
                                 tourney_size, mutation_rate,
                                 standard_deviation, ca_size, 
                                 connection_radius, ca_iterations, ca_repeat)
    
    # Get all individuals, ordered from the worst to the best
    individuals = evolve.get_individuals()
    
    # Send all the individuals to the analysis module
    for individual in individuals:
        analytics.add_individual(individual)
    
    best_individual = individuals[-1]
    best_fitness = best_individual.fitness
    
    log_iter(p_rewire, individuals)
    
    while not evolve.finished():
        # Execute a evolution step
        evolve.run()
        # Get all individuals, ordered from the worst to the best
        individuals = evolve.get_individuals()
        
        # Send all the individuals to the analysis module
        for individual in individuals:
            analytics.add_individual(individual)
        
        # Uptdate the best individual
        if individuals[-1].fitness > best_fitness:
            best_individual = individuals[-1]
            best_fitness = best_individual.fitness
        
        log_iter(p_rewire, individuals)
    
    # Print correlations between measures in the cellular automata evolved
    correlations = analytics.get_correlations()

    for group_name, group_data in correlations.iteritems():
        logging.info("%s" % group_name)
        
        for reference, reference_data in group_data.iteritems():
            logging.info("%s" % reference)
             
            for k, v in reference_data.iteritems():
                logging.info("\t%s\t%s" % (k, v))
            
            logging.info("")
        
        logging.info("")
    
    return best_individual, best_fitness

def parse_config(filename):
    with open(filename, "r") as config_f:
        lines = [l.split("#")[0].strip() for l in config_f.readlines()]
    
    config = dict([(i.strip() for i in l.split("=", 1)) for l in lines \
        if l != ""])
    
    keys_types = [("population_size", int), ("max_epochs", int),
                  ("tourney_size", int), ("mutation_rate", float),
                  ("standard_deviation", float),
                  ("ca_size", int), ("connection_radius", int), 
                  ("ca_iterations", int), ("ca_repeat", int),
                  ("k_history", int), ("save_executions", int)]
    
    return dict([(k, t(config[k])) for k, t in keys_types])

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate the evolution of "
        "small-world elementary cellular automata.")
    
    default_folder = "date-%s" % \
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--folder", dest="folder", action="store", type=str,
                        default=default_folder, help="Folder in which the "
                        "result of the simulation will be stored")
    parser.add_argument("--min-p", dest="min_p", action="store", type=float,
                        default=.0, help="The smallest amount of rewiring "
                        "that will be tested")
    parser.add_argument("--max-p", dest="max_p", action="store", type=float,
                        default=.1, help="The largest amount of rewiring "
                        "that will be tested")
    parser.add_argument("--step-p", dest="step_p", action="store", type=float,
                        default=.01, help="The increment in the amount of "
                        "rewiring in each test performed")
    parser.add_argument("config_file", action="store", type=str, 
                        help="File containing the configuration for the ECAs "
                        "to evaluated")
    
    return parser.parse_args()

def main():
    args = parse_args()
    ca_config = parse_config(args.config_file)
    
    aux.set_root_folder(args.folder)
    logging.basicConfig(filename=aux.get_root_folder() + '/execution.log', \
        filemode="a", level=logging.INFO)
    
    logging.info("==================================================")
    logging.info("Majority problem CA evolution log")
    logging.info("Contact godoy@dca.fee.unicamp.br for more information")
    logging.info("Date/time: %s" % time.asctime())
    logging.info("Arguments: %s" % sys.argv)
    logging.info("Configuration: %s" % ca_config.items())
    logging.info("")
    
    bests_fit_p = []
    p_rewire = args.min_p
    
    while p_rewire <= args.max_p:
        best_individual, best_fitness = analysis_p(p_rewire, **ca_config)
        bests_fit_p.append((p_rewire, best_individual.fitness, best_fitness))
        
        p_rewire += args.step_p
    
    logging.info("Fitness (uniform, constant) x p_rewire: %s" % bests_fit_p)

if __name__ == "__main__":
    main()
