
## HYPER BORG MO OPTIMISATION 
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" Main File to Run for MO optimization
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
import numpy as np
import json
import subprocess
import time



from hyperford.PyBorg.pyborg  import BorgMOEA
from platypus import  Problem, Real, NSGAII

# Import  the all the functions defined
from hyperford.optimise.MO_energy_function import MO_Opt_energy
from hyperford.model.model_functions import get_sampled_data


from hyperford.utils.parameters_check import get_parameter_constraints, validate_parameters
from hyperford.optimise.PostProcessor import MO_postplot, MO_scatterplot


# Define the  multi-objective optimization problem
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
        
        # Assume MO_Opt_energy is a function that returns the objectives
        objectives = MO_Opt_energy(self.Q, typet, conf, X_in, self.global_parameters, self.turbine_characteristics)
        return objectives


if __name__ == "__main__":
    
    # Make changes directly within the JSON file
    # After making changes, reload the JSON file to get the updated parameters
    subprocess.run(["python", "globalpars_JSON.py"])

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
        
    # Set the number of turbines for optimization
    numturbine = 3  # Example: optimization up to two turbine configurations
    
    problem_definition = MyMultiObjectiveProblem(numturbine, Q, global_parameters, turbine_characteristics)

    # Define the Platypus problem
    problem = Problem(2 + numturbine + 1, 2)
    problem.types[:2] = [Real(0.51, 3.49), Real(0.51, 3.49)]
    problem.types[2:] = [Real(1, 5)] + [Real(0.5, 20.0)] * numturbine
    problem.function = problem_definition.evaluate

    # Start the timer
    start_time = time.time()
    
    # Use the NSGA-II algorithm to solve the problem
    #algorithm = NSGAII(problem)
    
   # define and run the Borg algorithm for 10000 evaluations
    algorithm = BorgMOEA(problem, epsilons=0.001)
    algorithm.run(1000)

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    # Print the results
    for solution in algorithm.result:
       print(solution.objectives)
    
    # extract the solutions
    objectives = np.array([solution.objectives for solution in algorithm.result])
    X_opt = np.array([solution.variables for solution in algorithm.result])
    

    ## post processor, a table displaying the optimization results
    optimization_table = MO_postplot(objectives, X_opt)
    
    
    # Plot the results: Pareto Front
    MO_scatterplot(objectives[:, 0], objectives[:, 1])
    

    
    