## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""
""" 
Main File to Run
"""

# Import  the modules to be used from Library
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import statistics

# Import global parameters including site characteristics and streamflow records
import HP

# Import  the all the functions defined
from model_functions import *
from sim_energy_functions import *
from sim_operation_type import sim_config


# Run the model
typet =  1   # turbine type: 1 = Kaplan, 2 = Francis, 3= Pelton
conf = 2  # turbine config: 1 = single, 2 = dual, 3 = triple
D = 5 # as m
QL = 6 #as m3/s, design discharge of large turbine
QS = 5# as m3/s, design discharge of small turbine

P, AAE, OF = sim_config (typet, conf, D, QL, QS);
 # P: Daily power, AAE: Annual average energy, OF: Objective Function
