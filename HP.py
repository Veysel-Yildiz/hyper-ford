## HYPER SIMULATION & OPTIMISATION
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                   The University of Sheffield                            #
#                           March 2023                                     #
############################################################################
"""
""" 
Model inputs, project-based parameters  that change from one another 
"""


# Import  the modules to be used from Library
import numpy as np
import pandas as pd
#import math 
#import statistics

# Load the input data set
streamflow = np.loadtxt('input' + '/b_observed' + '.txt', dtype=float, delimiter=',')
MFD = 0.63 # the minimum environmental flow (m3/s)

Q = streamflow - MFD; # define discharge after environmental flow

Q[Q < 0] = 0 # Set negative values to zero using NumPy indexing

#####

hg = 117.3 # initial storage elevation (m), gross head
ht = 0  # tail water elevation, depth of outflow to stream (m)
L = 208 # the length of penstock (m)


cf = 0.15 #the so-called site factor, ranges between 0 and 1.5 (used for the cost of the civil works)
om  =  0.01 # ranges between 0.01 and 0.04,(used for maintenance and operation cost)
fxc  =  5 #the expropriation and other costs including transmission line

ep = 0.055 #electricity price in Turkey ($/kWh)
pt = 1500 #steel penstock price per ton ($/kWh)
i = 0.095 #the investment discount rate (or interest rate, %)
N = 49 # life time of the project (years)

# 
CRF = i*(1+i)**N/((1+i)**N-1) #capital recovery factor
tf  = 1 / ( 1 + i)**25 # 25 year of discount for electro-mechanic parts
#

ObjectiveF = 1 # Specify the objective function 1: NPV, 2: BC
operating_scheme = 1   #  1 = 1 small + identical, 2 = all identical, 3 = all varied

##################################################
# Define variables and interpolation function for the calculation of turbines efficiencies

# Load the efficiency curves 
Effcurves = pd.read_excel('input' + '/EffCurves' + '.xlsx', dtype=float)
EffCurves = np.array(Effcurves)

perc = EffCurves[:,0]

eff_kaplan = EffCurves[:,1] #Kaplan turbine efficiency

eff_francis = EffCurves[:,2] #Francis turbine efficiency

eff_pelton = EffCurves[:,3] #Pelton turbine efficiency

###############################################################################
###################################### do not change this parameters ############
##Model setup 

e = 0.45*10**(-4);        # epsilon (m) ; fiberglass e = 5*10^(-6) (m), concrete e = 1.8*10^(-4) (m)
v = 1.004*10**(-6);       # the kinematics viscosity of water (m2/s)
g = 9.81;                 # acceleration of gravity (m/s2)
ng = 0.98;                # generator-system efficiency
hr = 8760;                # total hours in a year
nf = [0.05, 0.33];        # specific spped range of francis turbine
nk = [0.19, 1.55];        # specific spped range of kaplan turbine
np = [0.005, 0.0612];     # specific spped range of pelton turbine
mf = 0.40;                # min francis turbine design flow rate
mk = 0.20;                # min kaplan turbine design flow rate
mp = 0.11;                # min pelton turbine design flow rate