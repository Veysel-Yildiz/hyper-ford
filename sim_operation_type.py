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
 D = penstock diameter
 QL = large tubine design discharge
 QS = Small tubine design discharge

"""
# Import  the all the functions defined
from sim_energy_functions import *


def sim_config( typet, conf, D, QL, QS): 
    
    if conf == 1: # 1 turbine
        [P,  AAE, OF] = Sim_calc_1 ( typet, conf, D, QL, QS);

        
    elif conf == 2: # 2 turbine
        [P,  AAE, OF] = Sim_calc_2 (typet, conf, D, QL, QS);
 
    
    elif conf == 3: # 3 turbine
        [P,  AAE, OF] = Sim_calc_3 ( typet, conf, D, QL, QS);   
    
    return P,  AAE, OF
