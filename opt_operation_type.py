
## HYPER OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           June 2024                                      #
############################################################################
"""
""" Return :
    
        OF : Objective Function  
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
from opt_energy_functions import *


def opt_config(x): 
    
 typet = round(x[0])
 conf = round(x[1])
 X =  np.array([x[2], x[3], x[4]]) # np.array([D, Q1, Q2])

    
 if conf == 1: # 1 turbine
  return  Opt_energy_single (typet, conf, X)

 
 else: #conf == 2: more than 1 turbine
  return  Opt_energy_OP (typet, conf, X)

   
 
