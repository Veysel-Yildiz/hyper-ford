## HYPER SIMULATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
"""  Return :
         AAE: Annual average energy
        NPV : Net Present Value in million USD
        BC  : Benefit to Cost Ratio
--------------------------------------
    Inputs :

global_parameters : structure of global variables
                hg: gross head(m)
                L : Penstock diameter (m)
               cf : site factor, used for the cost of the civil works
               om : maintenance and operation cost factor
              fxc : expropriation and other costs including transmission line
               ep :lectricity price in Turkey ($/kWh)
               pt : steel penstock price per ton ($/ton)
               ir : the investment discount rate (or interest rate)
                N : life time of the project (years)
 operating_scheme : turbine configuration setup 1 = 1 small + identical, 2 = all identical, 3 = all varied
    
Q : daily flow
typet : turbine type
conf : turbine configuration; single, dual, triple
X : array of design parameters;
X(1) : D, penstock diameter
X(2...) : tubine(s) design discharge

"""

# Import  the modules to be used from Library
import numpy as np
import math 

# Import  the all the functions defined
from hyperford.model.model_functions import moody, cost, operation_optimization, S_operation_optimization

def Sim_energy (Q, typet, conf, X, global_parameters,turbine_characteristics):

    # Extract parameters
    operating_scheme = global_parameters["operating_scheme"]
    case_specific = global_parameters["case_specific"]
    hg,  L, cf, om, fxc, ep, pt, ir, N = case_specific.values()
    e, hr, perc = global_parameters["e"], global_parameters["hr"], global_parameters["perc"]
    
    # Calculate derived parameters
    CRF = ir * (1 + ir)**N / ((1 + ir)**N - 1) #capital recovery factor
    tf = 1 / (1 + ir)**25 # 25 year of discount for electro-mechanic parts
    
    # Unpack the parameter values
    D = X[0]  # Diameter
    ed = e / D  # Relative roughness
    
    # Choose turbine characteristics
    kmin, var_name_cavitation, func_Eff = turbine_characteristics[typet]
    
    if conf == 1:  # Single operation
        Q_design = X[1]  # Design discharge
        
        # Calculate flow velocity and Reynolds number for design head
        V_d = 4 * Q_design / (np.pi * D**2)
        Re_d = V_d * D / 1e-6  # Kinematic viscosity ν = 1,002 · 10−6 m2∕s
        
        # Find the friction factor [-] for design head
        f_d = moody(ed, np.array([Re_d]))

        # Calculate head losses for design head
        hf_d = f_d * (L / D) * V_d**2 / (2 * 9.81) * 1.1  # 10% of local losses
        design_h = hg - hf_d  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L = 3000 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        ss_S = 214 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        
        if var_name_cavitation[1] <= ss_S or ss_L <= var_name_cavitation[0]:
            return -999999  # turbine type is not appropriate return
        
        # Calculate power
        q = np.minimum(Q, Q_design)  # Calculate q as the minimum of Q and Q_design
        n = np.interp(q / Q_design, perc, func_Eff)  # Interpolate values from func_Eff based on qt/Q_design ratio
        idx = q < kmin * Q_design  # Set qt and nrc to zero where qt is less than kmin * Q_design
        n[idx] = 0
        V = 4 * q / (np.pi * D**2)  # Flow velocity in the pipe
        Re = V * D / 1e-6  # Reynolds number
        f = moody(ed, Re)  # Friction factor
        hnet = hg - f * (L / D) * V**2 / (19.62 * 1.1)  # Head loss due to friction
        DailyPower = hnet * q * 9.81 * n * 0.98  # Power
        
    else:  # Dual and Triple turbine operation; operation optimization
        maxturbine = conf  # The number of turbines

        Qturbine = np.zeros(maxturbine) # Assign values based on the maximum number of turbines

        for i in range(1, maxturbine + 1):
            if operating_scheme == 1:
               Od = (i == 1) * X[1] + (i > 1) * X[2]
            elif operating_scheme == 2:
               Od = X[1]
            else:
               Od = X[i]
            Qturbine[i - 1] = Od
     
        Q_design = np.sum(Qturbine)  # Design discharge
        V_d = 4 * Q_design / (np.pi * D**2)  # Flow velocity for design head
        Re_d = V_d * D / 1e-6  # Reynolds number for design head
        f_d = moody(ed, np.array([Re_d]))  # Friction factor for design head
        hf_d = f_d * (L / D) * V_d**2 / (2 * 9.81) * 1.1  # Head losses for design head
        design_h = hg - hf_d  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L1 = 3000 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_S1 = 214 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_L2 = 3000 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75
        ss_S2 = 214 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75

        SSn = [1, 1]
        if var_name_cavitation[1] <= ss_S1 or ss_L1 <= var_name_cavitation[0]:
            SSn[0] = 0

        if var_name_cavitation[1] <= ss_S2 or ss_L2 <= var_name_cavitation[0]:
            SSn[1] = 0

        if sum(SSn) < 2:  # turbine type is not appropriate
           return -999999
        
        size_Q = len(Q)    # the size of time steps
        if size_Q < 1000:
          DailyPower = S_operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)
        else:
          DailyPower = operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)

    AAE = np.mean(DailyPower) * hr / 1e6  # Gwh Calculate average annual energy

    costP = cost(design_ic, design_h, typet, conf, D, global_parameters)

    # Unpack costs
    cost_em, cost_pen, cost_ph = costP[0], costP[1], costP[2]

    cost_cw = cf * (cost_pen + cost_em)  # (in dollars) civil + open channel + Tunnel cost

    Cost_other = cost_pen + cost_ph + cost_cw  # Determine total cost (with cavitation)

    T_cost = cost_em * (1 + tf) + Cost_other + fxc

    cost_OP = cost_em * om  # Operation and maintenance cost

    AR = AAE * ep * 0.98  # Annual Revenue in M dollars 2% will not be sold

    AC = CRF * T_cost + cost_OP  # Annual cost in M dollars

    NPV = (AR - AC) / CRF

    BC = AR / AC

    return AAE, NPV, BC
        
        
    


