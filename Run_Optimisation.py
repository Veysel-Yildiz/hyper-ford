## HYPER OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" 
Main File to Run for Optimization

Parameters for Differential Evolution (DE):
x(1), typet:  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)
x(2), conf: Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
x(3), D: Penstock diameter,
x(4) Od1: First turbine design docharge,
x(5) Od2: Second turbine design docharge,
x(6) Od3: Third turbine design docharge,
x(n) Odn: nth turbine design docharge,

"""

# Import  the modules to be used from Library
from scipy.optimize import  differential_evolution
import numpy as np
import pandas as pd

# Import  the all the functions defined
from opt_energy_functions import Opt_energy_single, Opt_energy_OP
from PostProcessor import postplot


def generate_bounds(numturbine):
    """
    Generate the bounds dynamically based on the number of turbines.
    First two is turbine type and number, third one is for D and rest is for turbines design discharge
    """
    base_bounds = [(0.51, 3.49), (0.51, 3.49), (1, 5)]
    turbine_bounds = [(0.5, 20)] * numturbine
    return base_bounds + turbine_bounds


def opt_config(x):
    """
    x, Parameters: Array of design variables including typet, conf, D, and Od values.
    Dynamically handle the X_in array based on the value of numturbine.
    
    Returns: The objective function value for the given configuration.
    """
    typet = round(x[0]) # Turbine type
    conf = round(x[1])  # Turbine configuration (single, dual, triple, etc.)
    X_in = np.array(x[2:2 + numturbine + 1])# Slicing input array for diameter and turbine design discharges
    
    if conf == 1: # 1 turbine
       return  Opt_energy_single (typet, 1, X_in)

    else: #conf == 2: more than 1 turbine
      return  Opt_energy_OP (typet, conf, X_in)


# Set the number of turbines for optimization
numturbine = 2  # Example: optimization up to two turbine configurations
bounds = generate_bounds(numturbine)


# Run the differential evolution optimization
result = differential_evolution(
    opt_config, 
    bounds, 
    maxiter=200, 
    popsize=20, 
    tol=0.001, 
    mutation=(0.5, 1), 
    recombination=0.7, 
    init='latinhypercube'
)


## post processor, a table displaying the optimization results
optimization_table = postplot(result)
