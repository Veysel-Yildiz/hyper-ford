# File: hyperford/optimise/Run_DE_optimisation.py

"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield, June 2024                 #
############################################################################
"""
""" Main File to Run for optimization
--------------------------------------
global_parameters (structure of global variables for turbine setup)
                 nf : specific speed range of francis turbine
                 nk : specific speed range of kaplan turbine
                 np : specific speed range of pelton turbine
                 mf : min francis turbine design flow rate
                 mk : min kaplan turbine design flow rate
                 mp : min pelton turbine design flow rate
         eff_kaplan : Kaplan turbine efficiency
        eff_francis : Francis turbine efficiency
         eff_pelton : Pelton turbine efficiency
--------------------------------------              
Return :
         OF : Objective function, Net Present Value (million USD) or  Benefit to Cost Ratio (-)
          X : Optimal design parameters;
          
X(1), typet :  Turbine type (1= Kaplan, 2= Francis, 3 = Pelton turbine)
 X(2), conf : Turbine configuration (1= Single, 2= Dual, 3 = Triple, ..nth Operation)
    X(3), D : Penstock diameter,
   X(4) Od1 : First turbine design discharge,
   X(5) Od2 : Second turbine design discharge,
   X(n) Odn : nth turbine design discharge,

"""

# Import necessary modules
from scipy.optimize import differential_evolution
import numpy as np
import json
import time
import logging
from hyperford.model.model_functions import get_sampled_data
from hyperford.optimise.PostProcessor import postplot, create_table
from hyperford.utils.parameters_check import get_parameter_constraints, validate_parameters
from hyperford.optimise.opt_energy_functions import Opt_energy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_bounds(numturbine):
    """
    Generate the bounds dynamically based on the number of turbines.
    First two are turbine type and configuration, third one is for D and rest are for turbine design discharges.
    """
    base_bounds = [(0.51, 3.49), (0.51, 3.49), (1, 5)]
    turbine_bounds = [(0.5, 20)] * numturbine
    return base_bounds + turbine_bounds

def opt_config(x, Q, global_parameters, turbine_characteristics):
    """
    Objective function to minimize/maximize based on Opt_energy

    Parameters: 
        x: Array of design variables including typet, conf, D, and Od values.
        Q: Discharge after environmental flow.
        global_parameters: Dictionary of global parameters for turbine setup.
        turbine_characteristics: Dictionary defining turbine characteristics and functions.

    Returns: 
        The objective function value for the given configuration.
    """
    typet = round(x[0])  # Turbine type
    conf = round(x[1])  # Turbine configuration (single, dual, triple, etc.)
    X_in = np.array(x[2:])  # Slicing input array for diameter and turbine design discharges

    return Opt_energy(Q, typet, conf, X_in, global_parameters, turbine_characteristics)

def main():
    
    logging.info("Starting the single-objective optimization...")

    try:
        with open('global_parameters.json', 'r') as json_file:
            global_parameters = json.load(json_file)
        logging.info("Loaded global parameters from JSON file.")
    except FileNotFoundError as e:
        logging.error(f"Error loading global_parameters.json: {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {e}")
        return

    parameter_constraints = get_parameter_constraints()

    try:
        validate_parameters(global_parameters, parameter_constraints)
        logging.info("Validated global parameters successfully.")
    except ValueError as e:
        logging.error(f"Parameter validation error: {e}")
        return

    try:
        streamflow = np.loadtxt('input/b_observed_long.txt', dtype=float, delimiter=',')
        logging.info("Loaded streamflow data from input file.")
    except FileNotFoundError as e:
        logging.error(f"Error loading streamflow data: {e}")
        return

    MFD = global_parameters["MFD"] # the minimum environmental flow (m3/s)
    use_sampling = True

    if use_sampling:
        sample_size = 10
        Sampled_streamflow = get_sampled_data(streamflow, sample_size)
        Q = np.maximum(Sampled_streamflow - MFD, 0)
    else:
        Q = np.maximum(streamflow - MFD, 0)
        

    # Define turbine characteristics and functions in a dictionary
    turbine_characteristics = {
        2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]),  # Francis turbine
        3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),   # Pelton turbine
        1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])   # Kaplan turbine type
    }

    # Set the number of turbines for optimization
    numturbine = 2  # Example: optimization up to two turbine configurations
    bounds = generate_bounds(numturbine)

    # Prompt user for number of evaluations and population size
    evaluations = int(input("Enter the number of evaluations (eg: 1000): "))
    population_size = int(input("Enter the size of population (eg: 100): "))

    # Start the timer
    start_time = time.time()

    # Run the differential evolution optimization
    result = differential_evolution(
        opt_config,
        bounds,
        args=(Q, global_parameters, turbine_characteristics),  # Pass additional arguments here
        maxiter=evaluations,
        popsize=population_size,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        init='latinhypercube'
    )

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Display the optimized results
    logging.info(f"Optimization result:\n{result}")

    # Post process and display results if needed
    optimization_table = postplot(result)
    # Save the optimization results to a text file
    create_table(optimization_table)
    logging.info("Optimization results saved to text file.")

if __name__ == "__main__":
    main()
