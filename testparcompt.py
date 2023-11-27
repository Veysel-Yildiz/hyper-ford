# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:16:16 2023

@author: vyildiz
"""


## HYPER OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""
""" 
Main File to Run for optimisation
"""

# Import  the modules to be used from Library
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import  differential_evolution
import multiprocessing

# Import global parameters including site characteristics and streamflow records
#import HP

# Import  the all the functions defined
#from model_functions import *
#from opt_energy_functions import *
from opt_operation_type import opt_config


### Parameter range for Differential Evolution (DE)
## turbine type, turbine configuration, D, QL, QS

# round (0.51-1.49) = 1, Kaplan turbine, 
# round (1.50-2.49) = 2, Francis turbine, 
# round (2.50-3.49) = 3, Pelton turbine, 

#Turbine tpype: 1 = Kaplan turbine,  2 = Francis turbine,  3 = Pelton turbine 
#Turbine configuration: 1 = Single,  2 = Dual,  3 = Triple Operation 

##Structure of DE optimisation

bounds = [(0.51,3.49), (0.51,3.49), (1, 5), (1, 20), (1, 10)]



# Define your objective function (opt_config) and bounds

# Number of CPU cores to use
num_cores = multiprocessing.cpu_count()

def optimize_parallel(seed):
    # Run differential_evolution with a unique random seed for each process
    result = differential_evolution(
        opt_config,
        bounds,
        maxiter=10,
        popsize=10,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        init='latinhypercube',
        seed=seed  # Use a unique seed for each process
    )
    return result

if __name__ == '__main__':
    # Create a pool of processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Generate unique seeds for each process
        seeds = range(num_cores)
        # Run optimization in parallel
        results = pool.map(optimize_parallel, seeds)

    # Find the best result among the parallel runs
    best_result = min(results, key=lambda x: x.fun)

# Now, best_result contains the best optimization result
