## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" Main File to Run for simulation

Parameters:
x(1), typet:  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)
x(2), conf: Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
x(3), D: Penstock diameter,
x(4) Od1: First turbine design docharge,
x(5) Od2: Second turbine design docharge,
x(n) Odn: nth turbine design docharge,

Return :
       AAE : Annual average energy
       NPV : Net Present Value in million USD
       BC  : Benefot to Cost Ratio
"""

import numpy as np
import pandas as pd
import json

from sim_energy_functions import Sim_energy

# Load the input data set
streamflow = np.loadtxt('input/b_observed.txt', dtype=float, delimiter=',')
MFD = 0.63  # the minimum environmental flow (m3/s)

# Define discharge after environmental flow
Q = np.maximum(streamflow - MFD, 0)

# Load the parameters from the JSON file
with open('global_parameters.json', 'r') as json_file:
    global_parameters = json.load(json_file)

# Define turbine characteristics and functions in a dictionary
turbine_characteristics = {
    2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]),# Francis turbine
    3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),# Pelton turbine
    1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])# Kaplan turbine type
}

# Setup the simulation model
typet = 2   # turbine type: 1 = Kaplan, 2 = Francis, 3 = Pelton
conf = 2    # turbine config: 1 = single, 2 = dual, 3 = triple
D = 2       # diameter (m)
Q1 = 5      # design discharge of first turbine (m3/s)
Q2 = 10     # design discharge of second turbine (m3/s)

X = np.array([D, Q1, Q2])

# Calculate simulation results
AAE, NPV, BC = Sim_energy(Q, typet, conf, X, global_parameters, turbine_characteristics)

print(AAE, NPV, BC)