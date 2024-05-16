## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           June 2024                                      #
############################################################################
"""
""" Return :
    
       AAE : Annual average energy
       NPV : Net Present Value in million USD
       BC  : Benefot to Cost Ratio
        
--------------------------------------
         
    Inputs :

          P : structure with variables used in calculation
          Q : daily flow
 ObjectiveF : objective function
      typet : turbine type
       conf : turbine configuration; single, dual, triple
          X : array of design parameters;
       X(1) : D, penstock diameter
    X(2...) : tubine(s) design discharge

"""
# Import  the all the functions defined
from sim_energy_functions import *


def sim_config( typet, conf, X): 
    
    if conf == 1: # 1 turbine
        [AAE, NPV, BC] = Sim_energy_single ( typet, conf, X);

        
    else: # 2 or more turbine
        [AAE,  NPV, BC] = Sim_energy_OP (typet, conf, X);
      
    return AAE,  NPV, BC
