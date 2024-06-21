# File: hyperford/optimise/Run_Borg_MO_optimisation.py

import numpy as np
import json
import time
import logging
from platypus import Problem, Real, NSGAII
from hyperford.PyBorg.pyborg import BorgMOEA
from hyperford.optimise.MO_energy_function import MO_Opt_energy
from hyperford.model.model_functions import get_sampled_data
from hyperford.utils.parameters_check import get_parameter_constraints, validate_parameters
from hyperford.optimise.PostProcessor import MO_postplot, MO_scatterplot, create_table

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyMultiObjectiveProblem:
    def __init__(self, numturbine, Q, global_parameters, turbine_characteristics):
        self.numturbine = numturbine
        self.Q = Q
        self.global_parameters = global_parameters
        self.turbine_characteristics = turbine_characteristics

    def evaluate(self, x):
        typet = np.round(x[0]).astype(int)  # Turbine type
        conf = np.round(x[1]).astype(int)  # Turbine configuration
        X_in = x[2:2 + self.numturbine + 1]
        objectives = MO_Opt_energy(self.Q, typet, conf, X_in, self.global_parameters, self.turbine_characteristics)
        return objectives

def main():
    
    # Ask the user for the number of evaluations
    evaluations = int(input("Enter the number of evaluations (eg: 1000): "))

    logging.info("Starting the Borg multi-objective optimization...")

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

    turbine_characteristics = {
        2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]), # Francis turbine
        3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),  # Pelton turbine
        1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])  # Kaplan turbine type
    }

    try:
        streamflow = np.loadtxt('input/b_observed_long.txt', dtype=float, delimiter=',')
        logging.info("Loaded streamflow data from input file.")
    except FileNotFoundError as e:
        logging.error(f"Error loading streamflow data: {e}")
        return

    MFD = 0.63  # the minimum environmental flow (m3/s)
    use_sampling = True

    if use_sampling:
        sample_size = 100
        Sampled_streamflow = get_sampled_data(streamflow, sample_size)
        Q = np.maximum(Sampled_streamflow - MFD, 0)
    else:
        Q = np.maximum(streamflow - MFD, 0)

    numturbine = 3  # Example: optimization up to three turbine configurations
    problem_definition = MyMultiObjectiveProblem(numturbine, Q, global_parameters, turbine_characteristics)

    problem = Problem(2 + numturbine + 1, 2)
    problem.types[:2] = [Real(0.51, 3.49), Real(0.51, 3.49)]
    problem.types[2:] = [Real(1, 5)] + [Real(0.5, 20.0)] * numturbine
    problem.function = problem_definition.evaluate

    start_time = time.time()

    algorithm = BorgMOEA(problem, epsilons=0.001)
    algorithm.run(evaluations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    objectives = np.array([solution.objectives for solution in algorithm.result])
    X_opt = np.array([solution.variables for solution in algorithm.result])


    logging.info("Optimization completed and results plotted.")
   
    optimization_table = MO_postplot(objectives, X_opt)
    MO_scatterplot(objectives[:, 0], objectives[:, 1])
    # Save the optimization results to a text file
    create_table(optimization_table)
    logging.info("Optimization results saved to text file.")

if __name__ == "__main__":
    main()
