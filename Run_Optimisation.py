
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

#result = differential_evolution(opt_config, bounds)

result = differential_evolution(opt_config, bounds, maxiter=10, 
        popsize=10, tol=0.001, mutation=(0.5, 1), recombination=0.7, init='latinhypercube')

#result = differential_evolution(opt_config, bounds, updating='deferred', workers=-1)
