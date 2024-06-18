## HYPER OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" Main File to Run for optimization
--------------------------------------
global_parameters (structure of global variables for turbine setup)
                 nf : specific spped range of francis turbine
                 nk : specific spped range of kaplan turbine
                 np : specific spped range of pelton turbine
                 mf : min francis turbine design flow rate
                 mk : min kaplan turbine design flow rate
                 mp : min pelton turbine design flow rate
         eff_kaplan : Kaplan turbine efficiency
        eff_francis : Francis turbine efficiency
         eff_pelton : Pelton turbine efficiency
--------------------------------------              
Return :
         OF : Objective function, Net Present Value (million USD) or  Benefot to Cost Ratio (-)
          X : Optimal design parameters;
          
X(1), typet :  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)
 X(2), conf : Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
    X(3), D : Penstock diameter,
   X(4) Od1 : First turbine design docharge,
   X(5) Od2 : Second turbine design docharge,
   X(n) Odn : nth turbine design docharge,

"""

# Import  the modules to be used from Library
from scipy.optimize import  differential_evolution
import numpy as np
import json
import time
import subprocess
#from numba import jit

# Import  the all the functions defined
from hyperford.optimise.opt_energy_functions import Opt_energy
from hyperford.model.model_functions import get_sampled_data
from hyperford.optimise.PostProcessor import postplot
from hyperford.utils.parameters_check import get_parameter_constraints, validate_parameters


def generate_bounds(numturbine):
    """
    Generate the bounds dynamically based on the number of turbines.
    First two is turbine type and number, third one is for D and rest is for turbines design discharge
    """
    base_bounds = [(0.51, 3.49), (0.51, 3.49), (1, 5)]
    turbine_bounds = [(0.5, 20)] * numturbine
    return base_bounds + turbine_bounds


def opt_config(x):
    """
    x, Parameters: Array of design variables including typet, conf, D, and Od values.
    Dynamically handle the X_in array based on the value of numturbine.
    
    Returns: The objective function value for the given configuration.
    """
    typet = round(x[0]) # Turbine type
    conf = round(x[1])  # Turbine configuration (single, dual, triple, etc.)
    X_in = np.array(x[2:2 + numturbine + 1])# Slicing input array for diameter and turbine design discharges
    
    return  Opt_energy (Q, typet, conf, X_in, global_parameters, turbine_characteristics)

if __name__ == "__main__":
    
    
    # Load the input data set
    streamflow = np.loadtxt('input/b_observed_long.txt', dtype=float, delimiter=',')
    MFD = 0.63  # the minimum environmental flow (m3/s)
    
    
    # Set this variable to True if you want to sample the streamflow data, False otherwise
    use_sampling = True

    if use_sampling:
        sample_size = 100
        # Get sampled streamflow data
        Sampled_streamflow = get_sampled_data(streamflow, sample_size)
        # Define discharge after environmental flow using sampled data
        Q = np.maximum(Sampled_streamflow - MFD, 0)
    else:
        # Define discharge after environmental flow using the entire dataset
        Q = np.maximum(streamflow - MFD, 0)
        
  
   
if __name__ == "__main__":
    # Make changes directly within the JSON file
    # After making changes, reload the JSON file to get the updated parameters
    subprocess.run(["python", "globalpars_JSON.py"])

    # Load the input data set
    streamflow = np.loadtxt('input/b_observed_long.txt', dtype=float, delimiter=',')
    MFD = 0.63  # the minimum environmental flow (m3/s)

    # Define discharge after environmental flow
    Q = np.maximum(streamflow - MFD, 0)

    # Load the parameters from the JSON file
    with open('global_parameters.json', 'r') as json_file:
        global_parameters = json.load(json_file)

    # Get the parameter constraints
    parameter_constraints = get_parameter_constraints()

    # Validate inputs
    validate_parameters(global_parameters, parameter_constraints)

    print("All inputs are valid.")


    # Define turbine characteristics and functions in a dictionary
    turbine_characteristics = {
        2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]),# Francis turbine
        3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),# Pelton turbine
        1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])# Kaplan turbine type
    }

    # Load the input data set
    streamflow = np.loadtxt('input/b_observed_long.txt', dtype=float, delimiter=',')
    MFD = 0.63  # the minimum environmental flow (m3/s)

    # Define discharge after environmental flow
    Q = np.maximum(streamflow - MFD, 0)

    # Load the parameters from the JSON file
    with open('global_parameters.json', 'r') as json_file:
        global_parameters = json.load(json_file)

    # Define turbine characteristics and functions in a dictionary
    turbine_characteristics = {
        2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]),# Francis turbine
        3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),# Pelton turbine
        1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])# Kaplan turbine type
    }

    # Set the number of turbines for optimization
    numturbine = 2  # Example: optimization up to two turbine configurations
    bounds = generate_bounds(numturbine)

    # Start the timer
    start_time = time.time()


     # Run the differential evolution optimization
    result = differential_evolution(
       opt_config, 
       bounds, 
       maxiter=10, 
       popsize=10, 
       tol=0.001, 
       mutation=(0.5, 1), 
       recombination=0.7, 
       init='latinhypercube'
       )

 
   ## post processor, a table displaying the optimization results
    optimization_table = postplot(result)

     # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")