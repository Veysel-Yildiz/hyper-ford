## HYPER SIMULATION & OPTIMISATION
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""

# Import  the modules to be used from Library
import scipy.optimize
import numpy as np
import math 
import statistics

# Import global parameters including site characteristics and streamflow records
import HP

##################################################COST#########################
def cost(design_ic, design_h, typet, conf, D, QL, QS):

 """ Return cost in million Dollars
 HP = structure with variables used in calculation
 design_ic = installed capacity
 design_h = design head
 type = turbine type
 conf = turbine configuration; single, dual, triple
 D = penstock diameter
 QL = large tubine design discharge
 QS = Small tubine design discharge

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
    cost_em = 2.927 * (design_ic/1000)**1.174 *(design_h)**-0.4933*1.1 # in $
    
 elif typet == 3: # pelton turbine cost
    cost_em = 1.984 * (design_ic/1000)**1.427 *(design_h)**-0.4808*1.1 # in $
    
 else: # Kaplan turbine cost
    if conf ==1:
            cost_em = 2.76 * (design_ic/1000)**0.5774 *(design_h)**-0.1193*1.1 #in $
    elif conf == 2:
            Pn1 =  design_ic* QL / (QS + QL)
            Pn2 =  design_ic* QS / (QS + QL)
            
            cost_em  = (2.76 * (Pn1/1000)**0.5774 *(design_h)**-0.1193*1.15 + 2.76 * (Pn2/1000)**0.5774 *(design_h)**-0.1193)*1.1
    elif conf == 3:
            Pn1 =  design_ic* QL / (QS + 2*QL)
            Pn2 =  design_ic* QS / (QS + 2*QL)
            cost_em  = (2*2.76 * (Pn1/1000)**0.5774 *(design_h)**-0.1193*1.15 + 2.76 * (Pn2/1000)**0.5774 *(design_h)**-0.1193)*1.1
 
 return cost_em , cost_pen,  cost_ph #tp,

##################################################MOODY########################
def moody(ed , Re):

 """ Return f, friction factor
 HP = structure with variables used in calculation
 ed= the relative roughness: epsilon / diameter.
 Re = the Reynolds number

 """
 if Re < 2000: # Laminar flow
    f = 64/Re
    
 elif Re>4000:
    f=1.325/(math.log(ed/3.7+5.74/(Re**0.9)))**2;
    
 else:
    Y3 = -0.86859 * math.log( ed/3.7 + 5.74 / (4000**0.9) );
    Y2 = ed/3.7 + 5.74 / (Re**0.9);
    FA = Y3**(-2);
    FB = FA * (2 - 0.00514215 / (Y2*Y3) );
    R = Re/2000;
    X1 = 7 * FA - FB;
    X2 = 0.128 - 17*FA + 2.5*FB;
    X3 = -0.128 + 13*FA - 2*FB;
    X4 = R * (0.032 - 3*FA + 0.5*FB);
    f = X1 + R*(X2 + R*(X3+X4));
 return f
    

