

## HYPER OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""
""" Return daily power production and objective functions for SINGLE, DUAL and TRIPLE operation mode
 HP = structure with variables used in calculation
 Q = daily flow
 ObjectiveF = objective function
 typet = turbine type
 conf = turbine configuration; single, dual, triple
 D = penstock diameter
 QL = large tubine design discharge
 QS = Small tubine design discharge

"""

# Import  the modules to be used from Library
import scipy.optimize
import numpy as np
import math 
import statistics
#import multiprocessing

# Import global parameters including site characteristics and streamflow records
import HP

# Import  the all the functions defined
from model_functions import *

## unpack global variables
ObjectiveF = HP.ObjectiveF
perc = HP.perc
L = HP.L
ve = HP.v
hg = HP.hg
ng = HP.ng
Q = HP.Q

maxT = len(Q)    # the size of time steps
P = np.empty((maxT)) #  create a new array for daily power
P[:] = np.NaN
 
## SINGLE turbine operation ###################################################
##################################################################SINGLE#######
def Opt_calc_1( typet, conf, D, QL, QS):

 Od = QL
  
 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

 Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Od / ( math.pi * D**2 )

 if V_d > 9 or V_d < 2.5:
    OF = -19999990*V_d
 else:

# choose turbine
  if typet == 2: # Francis turbine
     kmin = HP.mf;
     var_name_cavitation = HP.nf #specific speed range
     func_Eff = HP.eff_francis
    
  elif typet == 3: #Pelton turbine
     kmin = HP.mp;
     var_name_cavitation = HP.np #specific speed range
     func_Eff = HP.eff_pelton
    
  else:
     kmin = HP.mk
     var_name_cavitation = HP.nk #specific speed range
     func_Eff = HP.eff_kaplan

##head losses
  hf_d = f_d*(HP.L/D)*V_d**2/(2*HP.g)*1.1 # 10% of local losses
 #hl_d = HP.K_sum*V_d^2/(2*HP.g);

  design_h = HP.hg - hf_d;
  design_ic   = design_h * HP.g  * Od; # installed capacity

##  Now check the specific speeds of turbines 

  ss_L = 3000/60 * math.sqrt(Od)/(HP.g*design_h )**0.75;
  ss_S = 214/60 * math.sqrt(Od)/(HP.g*design_h  )**0.75;

  if var_name_cavitation[1] <= ss_S  or ss_L <= var_name_cavitation[0]:
      OF = -19999990*V_d  # turbine type is not apropriate return
  else:

# # ############ Now iterate over time % % % % % % % % % % % % % %
   for t in range(maxT): 
    
    # Check sum of Od1 and Od2
     q = min(Q[t] , Od);
    
     if  q < kmin*Od:
        # Turbine shut down
        P[t] = 0
        
     else:
        # Calculate the Reynolds number
        Re = 4 * q / ( math.pi * D * ve )
        
        # Find f, the friction factor [-]
        f  = moody ( ed , Re )
        
        #Claculate flow velocity in the pipe
        V = 4 * q / ( math.pi * D**2 )
        
        # Calculate the head loss due to friction in the penstock
        hf = f *(L/D)*V**2/(2*9.81)*1.1
        
        hnet = hg - hf
        
        # large francis/kaplan/pelton turbine efficiency
        ck = q/Od
        n = np.interp(ck, perc,func_Eff)
        
        P[t] = hnet * q * 9.81 * n * ng;
# # ############### End iterate over time % % % % % % % % % % % % % %

   costP = cost(design_ic, design_h, typet, conf, D, QL, QS)

      #Unpack costs
   cost_em = costP[0]; 
   cost_pen = costP[1];  
   cost_ph = costP[2]; #tp = costP(3);

   cost_cw = HP.cf * (cost_pen + cost_em ) #(in dollars) civil + open channel + Tunnel cost

   Cost_other = cost_pen + cost_ph + cost_cw #Determine total cost (with cavitation)

   T_cost = cost_em * (1+ HP.tf) + Cost_other + HP.fxc;

   cost_OP = cost_em * HP.om #operation and maintenance cost

   AAE = statistics.mean(P) * HP.hr/10**6 #Gwh Calculate average annual energy

   AR = AAE * HP.ep*0.97 # AnualRevenue in M dollars 3% will not be sold

   AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

   if ObjectiveF == 1:
        OF = (AR - AC ) / HP.CRF
   elif ObjectiveF == 2:
        OF = AR / AC
     
 return -OF
 

## DUAL turbine operation ###################################################
##################################################################DUAL#######
def Opt_calc_2 (typet, conf, D, QL, QS):
 

 if QS > QL: # check if capacity of large turbine is bigger than small turbine
    OF = -19999990*QS
 else:
     
  Od = QL +  QS

  ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

  Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
  f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
  V_d = 4 * Od / ( math.pi * D**2 )

  if V_d > 9 or V_d < 2.5:
    OF = -19999990*V_d
  else:

# choose turbine
   if typet == 2: # Francis turbine
     kmin = HP.mf
     var_name_cavitation = HP.nf #specific speed range
     func_Eff = HP.eff_francis
    
   elif typet == 3: #Pelton turbine
     kmin = HP.mp
     var_name_cavitation = HP.np #specific speed range
     func_Eff = HP.eff_pelton
    
   else:
     kmin = HP.mk
     var_name_cavitation = HP.nk #specific speed range
     func_Eff = HP.eff_kaplan

   ##head losses
   hf_d = f_d*(HP.L/D)*V_d**2/(2*HP.g)*1.1 # 10 % of local losses
   #hl_d = HP.K_sum*V_d^2/(2*HP.g);

   design_h = HP.hg - hf_d # design head
   design_ic   = design_h * HP.g  * Od # installed capacity

  ##  Now check the specific speeds of turbines 

   ss_L1 = 3000/60 * math.sqrt(QL)/(HP.g*design_h )**0.75
   ss_S1 = 214/60 * math.sqrt(QL)/(HP.g*design_h  )**0.75
 
   ss_L2 = 3000/60 * math.sqrt(QS)/(HP.g*design_h )**0.75
   ss_S2 = 214/60 * math.sqrt(QS)/(HP.g*design_h  )**0.75
 
   SSn = [1,1]
   if var_name_cavitation[1]  <= ss_S1  or ss_L1 <= var_name_cavitation[0]:
      SSn[0] = 0

   if var_name_cavitation[1]  <= ss_S2  or ss_L2 <= var_name_cavitation[0]:
      SSn[1] = 0

   if sum(SSn) < 2: # turbine type is not apropriate return
      OF = -19999990*V_d 
   else: 


# # ############ Now iterate over time % % % % % % % % % % % % % %
    for t in range(maxT): 
    
     # Check sum of Od1 and Od2
      q = min(Q[t] , Od);
    
     # Calculate the Reynolds number
      Re = 4 * q / ( math.pi * D * ve )
        
     # Find f, the friction factor [-]
      f  = moody ( ed , Re )
        
     #Claculate flow velocity in the pipe
      V = 4 * q / ( math.pi * D**2 )
        
     # Calculate the head loss due to friction in the penstock
      hf = f *(L/D)*V**2/(2*9.81)*1.1
    
      hnet = hg - hf
    
      if  q < kmin*QS: # Turbine shut down
        
          P[t] = 0
        
      elif q > kmin*QS  and q <=  QS: # only the small turbine in operation

         # small francis/kaplan/pelton turbine efficiency
          ck = q/QS
          n = np.interp(ck, perc,func_Eff)
        
          P[t] = hnet * q * 9.81 * n * ng
        
      elif q > QS  and q < QL +  kmin* QS: # only one turbine in operation, whihcever achives best production

         # large francis/kaplan/pelton turbine efficiency
          q1 = min(q , QL)
          ck = q1/QL
          n1 = np.interp(ck, perc,func_Eff)
          P1 = hnet * q1 * 9.81 * n1 * ng
        
         # small francis/kaplan/pelton turbine efficiency
          n2 = func_Eff[-1] 
          P2 = hnet * QS * 9.81 * n2 * ng       
        
          P[t] =  max( P1, P2 ) #[kW] maximum power produced
        
      else: # q >  QL +  kmin* QS: # both turbines in operation
         
         # large francis/kaplan/pelton turbine efficiency at full capacity
          n1 = func_Eff[-1] 
        
         # small francis/kaplan/pelton turbine efficiency
          ck  = min(1,(q - QL)/QS);
          n2 = np.interp(ck, perc,func_Eff)  
        
          P[t]  =  ( QL *n1 + (q - QL)*n2) * hnet * 9.81 * ng  
        
# # ############### End iterate over time % % % % % % % % % % % % % %

      costP = cost(design_ic, design_h, typet, conf, D, QL, QS);

    #Unpack costs
      cost_em = costP[0]
      cost_pen = costP[1] 
      cost_ph = costP[2] #tp = costP(3);

      cost_cw = HP.cf * (cost_pen + cost_em ) #(in dollars) civil + open channel + Tunnel cost

      Cost_other = cost_pen + cost_ph + cost_cw #Determine total cost (with cavitation)

      T_cost = cost_em * (1+ HP.tf) + Cost_other + HP.fxc;

      cost_OP = cost_em * HP.om #operation and maintenance cost

      AAE = statistics.mean(P) * HP.hr/10**6 #Gwh Calculate average annual energy

      AR = AAE * HP.ep*0.98 # AnualRevenue in M dollars 2% will not be sold

      AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

      if ObjectiveF == 1:
           OF = (AR - AC ) / HP.CRF
      elif ObjectiveF == 2:
           OF = AR / AC
     
 return -OF #, costP , AC, AR

##

## TRIPLE turbine operation ###################################################
##################################################################TRIPLE#######
def Opt_calc_3( typet, conf, D, QL, QS):

 Od = 2*QL +  QS

 if QS > QL: # check if the capacity of large turbine is bigger than small turbine
    OF = -19999990*QS
 else:
     
  ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

  Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
  f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
  V_d = 4 * Od / ( math.pi * D**2 )

  if V_d > 9 or V_d < 2.5:
    OF = -19999990*V_d
  else:

# choose turbine
   if typet == 2: # Francis turbine
     kmin = HP.mf
     var_name_cavitation = HP.nf #specific speed range
     func_Eff = HP.eff_francis
    
   elif typet == 3: #Pelton turbine
     kmin = HP.mp
     var_name_cavitation = HP.np #specific speed range
     func_Eff = HP.eff_pelton
    
   else:
     kmin = HP.mk
     var_name_cavitation = HP.nk #specific speed range
     func_Eff = HP.eff_kaplan

   ##head losses
   hf_d = f_d*(HP.L/D)*V_d**2/(2*HP.g)*1.1 # 10% of local losses
   #hl_d = HP.K_sum*V_d^2/(2*HP.g);

   design_h = HP.hg - hf_d # design head
   design_ic   = design_h * HP.g  * Od # installed capacity

  ##  Now check the specific speeds of turbines 

   ss_L1 = 3000/60 * math.sqrt(QL)/(HP.g*design_h )**0.75
   ss_S1 = 214/60 * math.sqrt(QL)/(HP.g*design_h  )**0.75
 
   ss_L2 = 3000/60 * math.sqrt(QS)/(HP.g*design_h )**0.75
   ss_S2 = 214/60 * math.sqrt(QS)/(HP.g*design_h  )**0.75
 
   SSn = [1,1]
   if var_name_cavitation[1]  <= ss_S1  or ss_L1 <= var_name_cavitation[0]:
      SSn[0] = 0

   if var_name_cavitation[1]  <= ss_S2  or ss_L2 <= var_name_cavitation[0]:
      SSn[1] = 0

   if sum(SSn) < 2: # turbine type is not apropriate return
      OF = -19999990*V_d 
   else: 
##

# # ############ Now iterate over time % % % % % % % % % % % % % %
    for t in range(maxT): 
    
    # Check sum of Od1 and Od2
     q = min(Q[t] , Od);
    
    # Calculate the Reynolds number
     Re = 4 * q / ( math.pi * D * ve )
        
    # Find f, the friction factor [-]
     f  = moody ( ed , Re )
        
    #Claculate flow velocity in the pipe
     V = 4 * q / ( math.pi * D**2 )
        
    # Calculate the head loss due to friction in the penstock
     hf = f *(L/D)*V**2/(2*9.81)*1.1
    
     hnet = hg - hf
    
     if  q < kmin*QS: # TurbineS shut down
        
         P[t] = 0
        
     elif q > kmin*QS  and q <=  QS: # only the small turbine in operation

        # small francis/kaplan/pelton turbine efficiency
         ck = q/QS
         n = np.interp(ck, perc,func_Eff)
        
         P[t] = hnet * q * 9.81 * n * ng
        
     elif q > QS  and q < QL +  kmin* QS: # only one turbine in operation, whihcever achives best production

        # large francis/kaplan/pelton turbine efficiency
         q1 = min(q , QL)
         ck = q1/QL
         n1 = np.interp(ck, perc,func_Eff)
         P1 = hnet * q1 * 9.81 * n1 * ng
        
         # small francis/kaplan/pelton turbine efficiency
         n2 = func_Eff[-1] 
         P2 = hnet * QS * 9.81 * n2 * ng       
        
         P[t] =  max( P1, P2 ) #[kW] maximum power produced
        
     elif q >  QL +  kmin* QS and q < QL + QS +  kmin* QL: # two turbines in operation
         
        # large francis/kaplan/pelton turbine efficiency at full capacity
         n1 = func_Eff[-1] 
        
         P1 = hnet * QL * 9.81 * n1 * ng;
         
        #check flow
         q2 = min(QS,(q - QL))
        
        # small francis/kaplan/pelton turbine efficiency
         ck = q2/QS
         n2 = np.interp(ck, perc,func_Eff)  
        
         P2 = hnet * q2 * 9.81 * n2 * ng;
        
         P[t]  =  P1 + P2 
        
     elif q >  QL + QS +  kmin* QL and q < 2*QL + kmin* QS: # three turbines in operation
         
        # large francis/kaplan/pelton turbine efficiency at full capacity
         n1 = func_Eff[-1] 
         P1 = hnet * QL * 9.81 * n1 * ng;
        
        # small francis/kaplan/pelton turbine efficiency at full capacity
         P2 = hnet * QS * 9.81 * n1 * ng;
        
        #update flow
         q3 = q - QL - QS
        
        # second large francis/kaplan/pelton turbine efficiency
         ck = q3/QL
         n3 = np.interp(ck, perc,func_Eff)  
        
         P3 = hnet * q3 * 9.81 * n3 * ng;
        
         P[t]  =  P1 + P2 + P3
        
     else:
        
        # two large francis/kaplan/pelton turbine efficiency at full capacity
         n1 = func_Eff[-1] 
         P12 = hnet * QL * 9.81 * n1 * ng;
        
        #update flow
         q3 = q - 2*QL
        
        # small francis/kaplan/pelton turbine efficiency
         ck = q3/QS
         n3 = np.interp(ck, perc,func_Eff)  
        
         P3 = hnet * q3 * 9.81 * n3 * ng;
        
         P[t]  = 2* P12 + P3
        
# # ############### End iterate over time % % % % % % % % % % % % % %

   costP = cost(design_ic, design_h, typet, conf, D, QL, QS);

#Unpack costs
   cost_em = costP[0]
   cost_pen = costP[1] 
   cost_ph = costP[2] #tp = costP(3);

   cost_cw = HP.cf * (cost_pen + cost_em ) #(in dollars) civil + open channel + Tunnel cost

   Cost_other = cost_pen + cost_ph + cost_cw #Determine total cost (with cavitation)

   T_cost = cost_em * (1+ HP.tf) + Cost_other + HP.fxc;

   cost_OP = cost_em * HP.om #operation and maintenance cost

   AAE = statistics.mean(P) * HP.hr/10**6 #Gwh Calculate average annual energy

   AR = AAE * HP.ep*0.98 # AnualRevenue in M dollars 2% will not be sold

   AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

   if ObjectiveF == 1:
        OF = (AR - AC ) / HP.CRF
   elif ObjectiveF == 2:
        OF = AR / AC
     
 return -OF

##
