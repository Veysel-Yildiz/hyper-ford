## HYPER MO OPTIMISATION 
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
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Import  the all the functions defined
from hyperford.optimise.MO_energy_function import MO_Opt_energy
from hyperford.model.model_functions import get_sampled_data


from hyperford.utils.parameters_check import get_parameter_constraints, validate_parameters
from hyperford.optimise.PostProcessor import MO_postplot, MO_scatterplot


# Define the problem class
class MyMultiObjectiveProblem(Problem):
    def __init__(self, numturbine, Q, global_parameters, turbine_characteristics):
        super().__init__(n_var=2 + numturbine + 1,
                         n_obj=2,  # Two objectives
                         n_constr=0,
                         xl=np.array([0.51, 0.51, 1] + [0.5] * numturbine),
                         xu=np.array([3.49, 3.49, 5] + [20.0] * numturbine))
        self.numturbine = numturbine
        self.Q = Q
        self.global_parameters = global_parameters
        self.turbine_characteristics = turbine_characteristics

    def _evaluate(self, x, out, *args, **kwargs):
        typet = np.round(x[:, 0]).astype(int)  # Turbine type
        conf = np.round(x[:, 1]).astype(int)  # Turbine configuration
        X_in = x[:, 2:2 + self.numturbine + 1]
        
        F1 = np.zeros(len(x))
        F2 = np.zeros(len(x))
        
        for i in range(len(x)):
            objectives = MO_Opt_energy(self.Q, typet[i], conf[i], X_in[i], self.global_parameters, self.turbine_characteristics)
            F1[i], F2[i] = objectives
        
        out["F"] = np.column_stack([F1, F2])


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

    # Create the problem instance
    problem = MyMultiObjectiveProblem(numturbine, Q, global_parameters, turbine_characteristics)

    # Define the algorithm
    algorithm = NSGA2(
        pop_size=50,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Start the timer
    start_time = time.time()

    # Perform the optimization
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=1,
                   verbose=True)

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Plot the results
    #Scatter().add(res.F).show()
   

    ## post processor, a table displaying the optimization results
    optimization_table = MO_postplot(res.F, res.X)
     
    # Plot the results: Pareto Front
    MO_scatterplot(res.F[:, 0], res.F[:, 1])
     

    