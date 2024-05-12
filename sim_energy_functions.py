## HYPER SIMULATION 
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
import numpy as np
import math 
import statistics

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
def Sim_calc_1(typet, conf, D, QL, QS):

 Od = QL
  
 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

 Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Od / ( math.pi * D**2 )

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
    SS = 0
 else:
    SS = 1

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

 costP = cost(design_ic, design_h, typet, conf, D, QL, QS);

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
     
 return P, AAE, OF


## DUAL turbine operation ###################################################
##################################################################DUAL#######
def Sim_calc_2(typet, conf, D, QL, QS):

 Od = QL +  QS

 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

 Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Od / ( math.pi * D**2 )

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

 if sum(SSn) == 2:
    SS = 1
 else: 
    SS = 0
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

 AR = AAE * HP.ep*0.98 # AnualRevenue in M dollars 3% will not be sold

 AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

 if ObjectiveF == 1:
     OF = (AR - AC ) / HP.CRF
 elif ObjectiveF == 2:
     OF = AR / AC
     
 return P, AAE, OF

##





## DUAL turbine operation ###################################################
##################################################################DUAL#######
def Sim_energy_OP(typet, conf, X):
 
    
 maxturbine = conf; # the number of turbines 
 
 D = X[0] # diameter
 
 # Handle the opscheme and turbine assignments
 operating_scheme = HP.operating_scheme  # 1 = 1 small + identical, 2 = all identical, 3 = all varied

# Assign values based on the maximum number of turbines
 Qturbine = np.zeros(maxturbine)

 for i in range(1, maxturbine + 1):
    if operating_scheme == 1:
        Od = (i == 1) * X[1] + (i > 1) * X[2]
    elif opscheme == 2:
        Od = X[1]
    else:
        Od = X[i]
    
    Qturbine[i - 1] = Od

 Od1 = Qturbine[0]
 Od2 = Qturbine[1]

 Q_design = np.sum(Qturbine)  # find design discharge

 
 # choose turbine characteristics
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
 

 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

 Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Od / ( math.pi * D**2 )


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

 if sum(SSn) == 2:
    SS = 1
 else: 
    SS = 0
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

 AR = AAE * HP.ep*0.98 # AnualRevenue in M dollars 3% will not be sold

 AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

 if ObjectiveF == 1:
     OF = (AR - AC ) / HP.CRF
 elif ObjectiveF == 2:
     OF = AR / AC
     
 return P, AAE, OF

#


## TRIPLE turbine operation ###################################################
##################################################################TRIPLE#######
def Sim_calc_3(typet, conf, D, QL, QS):

 Od = 2*QL +  QS

 ed = HP.e / D # calculate the relative roughness: epsilon / diameter.

 #design head ------------------------------------------

 Re_d = 4 * Od / ( math.pi * D * HP.v ) #Calculate the Reynolds number for design head

# Find f, the friction factor [-] for design head
 f_d = moody ( ed , Re_d )

# Claculate flow velocity in the pipe for design head
 V_d = 4 * Od / ( math.pi * D**2 )

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

 if sum(SSn) == 2:
    SS = 1
 else: 
    SS = 0
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

 AR = AAE * HP.ep*0.98 # AnualRevenue in M dollars 3% will not be sold

 AC = HP.CRF * T_cost + cost_OP; # Anual cost in M dollars

 if ObjectiveF == 1:
     OF = (AR - AC ) / HP.CRF
 elif ObjectiveF == 2:
     OF = AR / AC
     
 return P, AAE, OF

##
