
This repository contains Python code of HYPER HYdroPowER or HYPER, which uses a daily time step to simulate the technical performance, energy production, maintenance and operational costs, and economic profit of a RoR plant in response to a suite of different design and construction variables and record of river flows. The model includes a evolutionary algorithm that enables the user to maximize the RoR plant's power production or net economic profit by optimizing (among others) the penstock diameter, and the type (Kaplan, Francis, Pelton) design flow, and configuration (single/dual/triple) of the turbine system. What is more, it also simulates a predefined design.
The toolbox introduced in the paper by  V. Yildiz, J. Vrugt, "A toolbox for the optimal design of run-of-river hydropower plants" Environmental Modelling & Software
The library is built exclusively using Python code; It consists of following files. 

Contents:

`HP.py`: This is the main input file that contains global parameters both for optimisation and simulation.

`Run_Simulation.py`: This is the main file to run to simulate enrgy production based on predifined design parameters.


`sim_energy_functions.py`: This file contains mainly three functions that return daily power production and objective functions for SINGLE, DUAL and TRIPLE operation mode for simulation.

`model_functions.py`: This file contains two functions; (i) return the cost of a project in USD, (ii) returns the friction factor to calculate hydraulic losses. 

`Run_Optimisation.py`: This is the main file to run to optimise a design of a project. 


`opt_energy_functions.py`: This file contains mainly three functions that return daily power production and objective functions for SINGLE, DUAL and TRIPLE operation mode for simulation.

`model_functions.py`: This file contains two functions; (i) return the cost of a project in USD, (ii) returns the friction factor to calculate hydraulic losses. 




