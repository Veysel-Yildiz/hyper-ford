
# HYPER: HYdroPowER Simulation and Optimization Toolbox

Welcome to the HYPER repository! This repository contains Python code for HYPER (HYdroPowER), a comprehensive tool designed to simulate and optimize the performance of run-of-river (RoR) hydropower plants. The toolbox is built exclusively using Python and introduced in the paper by V. Yildiz and J. Vrugt, "A toolbox for the optimal design of run-of-river hydropower plants" published in Environmental Modelling &amp; Software.

## Overview 

HYPER uses a daily time step to simulate various aspects of RoR hydropower plants, including:
Technical performance
Energy production
Maintenance and operational costs
Economic profit

The toolbox accounts for different design and construction variables and utilizes historical river flow records. It also includes an evolutionary algorithm to optimize various design parameters, such as:
Penstock diameter
Turbine type (Kaplan, Francis, Pelton)
Turbine design flow
Turbine configuration (single or multiple)

Additionally, HYPER allows for the simulation of predefined designs.

## Contents

global_parameters.json: Contains global parameters for both optimization and simulation.

Run_Simulation.py: Main script to simulate energy production based on predefined design parameters.

sim_energy_functions.py: Includes functions for daily power production and objective functions for single and multiple operation modes in simulations.

model_functions.py: Contains all required sub-functions for the simulation and optimization processes.

Run_Optimisation.py: Main script to optimize the design of an RoR project.

opt_energy_functions.py: Includes functions for daily power production and objective functions for single and multiple operation modes in optimization.

## Getting Started

Load Input Data: Ensure you have the necessary input data files, such as b_observed.txt, containing river flow records.

Global Parameters: Modify the global_parameters.json file to suit your specific project requirements.

Run Simulation: Use Run_Simulation.py to simulate energy production based on predefined design parameters.

Run Optimization: Use Run_Optimisation.py to find the optimal design for your RoR hydropower plant.

## Usage

To run a simulation: python Run_Simulation.py

To run an optimization:python Run_Optimisation.py

## Install as package

To install the project as a python package:
1. Download project from Github
2. Navigate to Project directory with command line interface
3. Create a new venv `py -m venv .venv`
4. Activate environment
    1. Windows: `.venv\Scripts\activate`
    2. macOS / Linux: `source .venv/Scripts/activate`
5. Install package: `py -m pip install .`
    1. If you are developing on the package you may wish to do an editable install: `py -m pip install -e .`


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The development of this toolbox was introduced in the paper by V. Yildiz and J. Vrugt, "A toolbox for the optimal design of run-of-river hydropower plants," published in Environmental Modelling &amp; Software.

