## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""
""" Return P: Daily power, AAE: Annual average energy, OF: Objective Function 
 HP = structure with variables used in calculation
 Q = daily flow
 ObjectiveF = objective function
 typet = turbine type
 conf = turbine configuration; single, dual, triple
 X = array of design parameters;
 X(1) = D, penstock diameter
 X(2...) = tubine(s) design discharge

"""
# Import  the all the functions defined
from sim_energy_functions import *


def sim_config( typet, conf, X): 
    
    if conf == 1: # 1 turbine
        [P,  AAE, OF] = Sim_energy_single ( typet, conf, X);

        
    else: # 2 or more turbine
        [P,  AAE, OF] = Sim_energy_OP (typet, conf, X);
      
    return P,  AAE, OF
