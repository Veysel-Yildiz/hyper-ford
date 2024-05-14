## HYPER SIMULATION & OPTIMISATION
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           June 2024                                      #
############################################################################
"""

# Import  the modules to be used from Library
import scipy.optimize
import numpy as np
import math 
import statistics
from itertools import combinations

# Import global parameters including site characteristics and streamflow records
import HP

##################################################COST#########################
def cost(design_ic, design_h, typet, conf, D):

 """ Return:

     cost_em : Electro-mechanic (turbine) cost in million USD
    cost_pen : Penstock cost in million USD
     cost_ph : Powerhouse cost in million USD
 
--------------------------------------
    
   Inputs :

        HP : structure with variables used in calculation
 design_ic : installed capacity
  design_h : design head
     typet : turbine type
      conf : turbine configuration; single, dual, triple
         D : penstock diameter
 """
 tp  = 8.4/1000*D + 0.002 # Thickness of the pipe  [m]

 #tp1 = (1.2 * HP.hd * D / ( 20* 1.1) +2)*0.1; #t = (head + water hammer head (20%))*D / The working stress of the steel * 2
 #tp2 = (D + 0.8) / 4; % min thickness in cm
 #tp = max(tp1, tp2)/100;% min thickness in m

 cost_pen = math.pi * tp * D * HP.L * 7.874 * HP.pt/10**6;

 #Calculate the cost of power house (in M dollars)
 cost_ph = 200 * (design_ic/1000)**-0.301  * design_ic/10**6;

 # Calculate the cost of power house (in Mdollars)
 #cost_ph = HP.power*100/10**6;
 #cost_ph = HP.power / 10000;

 #Switch among the different turbine combinations
 
 if typet == 2: # Francis turbine cost
    cost_em = 2.927 * (design_ic/1000)**1.174 *(design_h)**-0.4933*1.1 * (1 + (conf-1)*(conf-2)*0.03) # in $
    
 elif typet == 3: # pelton turbine cost
    cost_em = 1.984 * (design_ic/1000)**1.427 *(design_h)**-0.4808*1.1 * (1 + (conf-1)*(conf-2)*0.03) # in $
    
 else: # Kaplan turbine cost

    cost_em = 2.76 * (design_ic/1000)**0.5774 *(design_h)**-0.1193*1.1 * (1 + (conf-1)*(conf-2)*0.03) # in $

 return cost_em , np.array([cost_pen]),  cost_ph #tp,

##################################################MOODY########################
def moody(ed , Re):

 """ Return f, friction factor

--------------------------------------
 
  Inputs:

    HP : structure with variables used in calculation
    ed : the relative roughness: epsilon / diameter.
    Re : the Reynolds number

 """
 f = np.zeros_like(Re)

 # Find the indices for Laminar, Transitional and Turbulent flow regimes
 
 LamR = np.where((0 < Re) & (Re < 2000))
 LamT = np.where(Re > 4000)
 LamTrans = np.where((2000 < Re) & (Re < 4000))

 f[LamR] = 64 / Re[LamR]

 # Calculate friction factor for Turbulent flow using the Colebrook-White approximation
 f[LamT] = 1.325 / (np.log(ed / 3.7 + 5.74 / (Re[LamT] ** 0.9)) ** 2)
  
 Y3 = -0.86859 * np.log(ed / 3.7 + 5.74 / (4000 ** 0.9))
 Y2 = ed / 3.7 + 5.74 / (Re[LamTrans] ** 0.9)
 FA = Y3 ** (-2)
 FB = FA * (2 - 0.00514215 / (Y2 * Y3))
 R = Re[LamTrans] / 2000
 X1 = 7 * FA - FB
 X2 = 0.128 - 17 * FA + 2.5 * FB
 X3 = -0.128 + 13 * FA - 2 * FB
 X4 = R * (0.032 - 3 * FA + 0.5 * FB)
 f[LamTrans] = X1 + R * (X2 + R * (X3 + X4))

 return f
    
################################################## operation optimization  #########################

def inflow_allocation(nr, Od, q_inc, kmin, perc, func_Eff):
     
 """ Return:

          qt : Turbine inflow for each incremental step.
      Eff_qi : Efficiency and inflow multiplication for energy calculation.
         nrc : Turbine running capacity as a ratio
 
--------------------------------------
           
    Inputs:

          nr : Turbine random sampled allocated discharge.
          Od : Turbine design discharge.
       q_inc : Incremental flow steps between turbine min and  max (design) discharge.
        kmin : Minimum turbine discharge to operate.
        perc : Efficiency percentile.
    func_Eff : Efficiency curve.

 """

 # Multiply each row of nr by the corresponding element of q_inc
 nrc = nr * q_inc

 # Calculate qt as the minimum of nrc and Od
 qt = np.minimum(nrc, Od)

 # Interpolate values from func_Eff based on qt/Od ratio
 Daily_Efficiency = np.interp(qt / Od, perc, func_Eff)

 # Set qt and nrc to zero where qt is less than kmin * Od
 idx = qt < kmin * Od
 qt[idx] = 0
 nrc[idx] = 0

 # Calculate np as the product of Efficiency and qt
 Eff_qi = Daily_Efficiency * qt

 return qt, Eff_qi, nrc


################################################## possible combinations #########################

def generate_patterns(maxturbine):
    # Function to generate the required combinations
    
    """ Return pattern, all possible combinations of turbines at full capacity
 
    --------------------------------------
    
    Inputs:

    maxturbine : Number of turbine

    """
    
    patterns = [] # Initialize an empty list to store patterns
    
    # Generate all possible patterns of 1s and 0s for the given maxturbine, 
    for num_ones in range(1, maxturbine + 1): 
        
        # Iterate over combinations of indices for placing 1s
        for comb in combinations(range(maxturbine), num_ones):
            
            pattern = [0] * maxturbine # Initialize pattern with all 0s
            for index in comb: # Set indices corresponding to 1s
                pattern[index] = 1
            patterns.append(pattern) # Add the pattern to the list
    return patterns