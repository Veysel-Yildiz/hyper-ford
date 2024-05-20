## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" Main File to Run for simulation

Parameters to be used for simulation:
    
x(1), typet:  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)
x(2), conf: Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
x(3), D: Penstock diameter,
x(4) Od1: First turbine design docharge,
x(5) Od2: Second turbine design docharge,
x(n) Odn: nth turbine design docharge,
--------------------------------------
global_parameters (structure of global variables for turbine setup);
                 nf : specific spped range of francis turbine
                 nk : specific spped range of kaplan turbine
                 np : specific spped range of pelton turbine
                 mf : min francis turbine design flow rate
                 mk : min kaplan turbine design flow rate
                 mp : min pelton turbine design flow rate
         eff_kaplan : Kaplan turbine efficiency
        eff_francis : Francis turbine efficiency
         eff_pelton : Pelton turbine efficiency
--------------------------------------              
Return :
       AAE : Annual average energy (GWh)
       NPV : Net Present Value (million USD)
       BC  : Benefot to Cost Ratio (-)
"""

import numpy as np
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

# Setup the model
typet = 2           # turbine type: 1 = Kaplan, 2 = Francis, 3 = Pelton
D = 2               # Define diameter (m) 
Qdesign = [5, 10, 15]  # Add the design discharge values here

conf = len(Qdesign) # Set conf to the size of Qdesign

X = np.array([D] + Qdesign) # Update X array with D and Qdesign values

# Calculate simulation results
AAE, NPV, BC = Sim_energy(Q, typet, conf, X, global_parameters, turbine_characteristics)

print(f"AAE: {AAE} GWh, NPV: {NPV} million USD, BC: {BC}")