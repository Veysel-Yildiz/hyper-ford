
## HYPER OPTIMISATION 
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
from opt_energy_functions import *


def opt_config(x): 
    
 if round(x[1]) == 1: #conf == 1: 1 turbine
  return  Opt_calc_1 ( round(x[0]), round(x[1]), x[2], x[3], x[4])  

 
 if round(x[1]) == 2: #conf == 2: 2 turbine
  return  Opt_calc_2 ( round(x[0]), round(x[1]), x[2], x[3], x[4])

   
 if round(x[1]) == 3: #conf == 3: 3 turbine
  return  Opt_calc_3 ( round(x[0]), round(x[1]), x[2], x[3], x[4])

    