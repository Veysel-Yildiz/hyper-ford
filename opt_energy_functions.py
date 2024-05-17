## HYPER OPTIMIZATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
"""  Return :
         OF : Objective Function  
--------------------------------------
    Inputs :

         HP : structure of global variables
          Q : daily flow
 ObjectiveF : objective function
      typet : turbine type
       conf : turbine configuration; single, dual, triple
          X : array of design parameters;
       X(1) : D, penstock diameter
    X(2...) : tubine(s) design discharge

"""

# Import  the modules to be used from Library
import numpy as np
import math 

#import multiprocessing

# Import global parameters including site characteristics and streamflow records
import HP

# Import  the all the functions defined
from model_functions import moody, cost, operation_optimization

## unpack global variables
ObjectiveF = HP.ObjectiveF
perc = HP.perc
hg = HP.hg
L = HP.L
Q = HP.Q

maxT = len(Q)    # the size of time steps

## SINGLE turbine operation ###################################################

def Opt_energy_single(typet, conf, X):
 
 #Unpack the parameter values
 D = X[0] # diameter
 Q_design = X[1] # design discharge

  #design head ------------------------------------------

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Q_design / ( np.pi * D**2 )
 
 penalty = 19999990

 if V_d > 9 or V_d < 2.5:
     return penalty * V_d
 
 #Calculate the Reynolds number for design head
 Re_d =  V_d * D / 10**-6  # kinematic viscosity ν = 1,002 · 10−6 m2∕s

 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , np.array([Re_d]) )
 
 # choose turbine characteristics
 kmin, var_name_cavitation, func_Eff = HP.turbine_characteristics[typet]

 ##head losses
 hf_d = f_d * (HP.L / D) * V_d ** 2 / (2 * 9.81) * 1.1  # 10% of local losses
 design_h = HP.hg - hf_d  # design head

 design_ic = design_h * 9.81 * Q_design  # installed capacity 

 ##  Now check the specific speeds of turbines
 ss_L = 3000 / 60 * math.sqrt(Q_design) / (9.81 * design_h) ** 0.75
 ss_S = 214 / 60 * math.sqrt(Q_design) / (9.81 * design_h) ** 0.75

 if var_name_cavitation[1] <= ss_S or ss_L <= var_name_cavitation[0]:
    return penalty * V_d  # turbine type is not appropriate return

 # Calculate q as the minimum of Q and Od
 q = np.minimum(Q, Q_design) 


 # Interpolate values from func_Eff based on qt/Od ratio
 n = np.interp(q / Q_design, perc, func_Eff)
 
 
 # Set qt and nrc to zero where qt is less than kmin * Od
 idx = q < kmin * Q_design
 n[idx] = 0

 # Calculate flow velocity in the pipe
 V = 4 * q / (np.pi * D**2)
 
 #Calculate the Reynolds number 
 Re =  V * D / 10**-6  # kinematic viscosity ν = 1,002 · 10−6 m2∕s
 
 # Find f, the friction factor [-]
 f = moody(ed, Re)

 # Calculate the head loss due to friction in the penstock
 hnet = hg - f * (L / D) * V**2 / 19.62 * 1.1

 # Calculate power
 DailyPower = hnet * q * 9.81 * n * 0.98
 
 AAE = np.mean(DailyPower) * HP.hr / 10 ** 6  # Gwh Calculate average annual energy

 costP = cost(design_ic, design_h, typet, conf, D)

 # Unpack costs
 cost_em, cost_pen, cost_ph = costP[0], costP[1], costP[2]

 cost_cw = HP.cf * (cost_pen + cost_em)
 Cost_other = cost_pen + cost_ph + cost_cw

 T_cost = cost_em * (1 + HP.tf) + Cost_other + HP.fxc
 cost_OP = cost_em * HP.om
 AR = AAE * HP.ep * 0.98

 AC = HP.CRF * T_cost + cost_OP

 if ObjectiveF == 1:
     OF = (AR - AC) / HP.CRF
 elif ObjectiveF == 2:
     OF = AR / AC

 return -OF
 

## DUAL turbine operation ###################################################
##################################################################DUAL#######
def Opt_energy_OP(typet, conf, X):
 

 maxturbine = conf; # the number of turbines 
 
 D = X[0] # diameter
 
 # Handle the opscheme and turbine assignments
 operating_scheme = HP.operating_scheme  # 1 = 1 small + identical, 2 = all identical, 3 = all varied

# Assign values based on the maximum number of turbines
 Qturbine = np.zeros(maxturbine)

 for i in range(1, maxturbine + 1):
    if operating_scheme == 1:
        Od = (i == 1) * X[1] + (i > 1) * X[2]
    elif operating_scheme == 2:
        Od = X[1]
    else:
        Od = X[i]
    
    Qturbine[i - 1] = Od

 Od1 = Qturbine[0]
 Od2 = Qturbine[1]
 
 # Design head calculation
 Q_design = np.sum(Qturbine) # find design discharge
 
 # Claculate flow velocity in the pipe for design head
 V_d = 4 * Q_design / ( np.pi * D**2 )
 
 penalty = 19999990

 if V_d > 9 or V_d < 2.5:
     return penalty * V_d
 
 #Calculate the Reynolds number for design head
 Re_d =  V_d * D / 10**-6  # kinematic viscosity ν = 1,002 · 10−6 m2∕s

 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , np.array([Re_d]) )
 

 # choose turbine characteristics
 kmin, var_name_cavitation, func_Eff = HP.turbine_characteristics[typet]

# head losses
 hf_d = f_d * (HP.L / D) * V_d ** 2 / (2 * 9.81) * 1.1  # 10% of local losses,#hl_d = HP.K_sum*V_d^2/(2*HP.g);
 
 design_h = HP.hg - hf_d  # design head
 
 design_ic = design_h * 9.81 * Q_design  # installed capacity

 # Now check the specific speeds of turbines
 ss_L1 = 3000 / 60 * math.sqrt(Od1) / (9.81 * design_h) ** 0.75
 ss_S1 = 214 / 60 * math.sqrt(Od1) / (9.81 * design_h) ** 0.75
 ss_L2 = 3000 / 60 * math.sqrt(Od2) / (9.81 * design_h) ** 0.75
 ss_S2 = 214 / 60 * math.sqrt(Od2) / (9.81 * design_h) ** 0.75

 SSn = [1, 1]
 if var_name_cavitation[1] <= ss_S1 or ss_L1 <= var_name_cavitation[0]:
     SSn[0] = 0
 if var_name_cavitation[1] <= ss_S2 or ss_L2 <= var_name_cavitation[0]:
      SSn[1] = 0

 if sum(SSn) < 2:  # turbine type is not appropriate
     return penalty * V_d 

 DailyPower = operation_optimization(maxturbine, Qturbine, Q_design, D, kmin, func_Eff)
 
 AAE = np.mean(DailyPower) * HP.hr / 10 ** 6  # Gwh Calculate average annual energy

 costP = cost(design_ic, design_h, typet, conf, D)

 # Unpack costs
 cost_em, cost_pen, cost_ph = costP[0], costP[1], costP[2]

 cost_cw = HP.cf * (cost_pen + cost_em)
 Cost_other = cost_pen + cost_ph + cost_cw

 T_cost = cost_em * (1 + HP.tf) + Cost_other + HP.fxc
 cost_OP = cost_em * HP.om
 AR = AAE * HP.ep * 0.98

 AC = HP.CRF * T_cost + cost_OP

 if ObjectiveF == 1:
     OF = (AR - AC) / HP.CRF
     
 elif ObjectiveF == 2:
     OF = AR / AC

 return -OF 

##

