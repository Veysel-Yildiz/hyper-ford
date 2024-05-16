## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield                            #
#                           June 2024                                      #
############################################################################
"""
""" Main File to Run for simulation

Parameters:
x(1), typet:  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)

x(2), conf: Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
 
x(3), D: Penstock diameter,

x(4) Od1: First turbine design docharge,
x(5) Od2: Second turbine design docharge,
...
x(n) Odn: nth turbine design docharge,

Return :
       AAE : Annual average energy
       NPV : Net Present Value in million USD
       BC  : Benefot to Cost Ratio
"""

# Import  the modules to be used from Library
import numpy as np

# Import  the all the functions defined
from sim_energy_functions import Sim_energy_single, Sim_energy_OP

def sim_config( typet, conf, X): 
    if conf == 1: # 1 turbine
        [AAE, NPV, BC] = Sim_energy_single ( typet, conf, X);
    else: # 2 or more turbine
        [AAE,  NPV, BC] = Sim_energy_OP (typet, conf, X); 
    return AAE,  NPV, BC

# Setup the model
typet =  2   # turbine type: 1 = Kaplan, 2 = Francis, 3= Pelton
conf = 2  # turbine config: 1 = single, 2 = dual, 3 = triple
D = 2 # as m
Q1 = 5 #as m3/s, design discharge of first turbine
Q2 = 10# as m3/s, design discharge of second turbine

X =  np.array([D, Q1, Q2])

AAE,  NPV, BC = sim_config (typet, conf, X);

print(AAE,  NPV, BC)