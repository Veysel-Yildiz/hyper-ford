
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

### Prerequisites

1. **Project Folder:** Create a project folder on your desktop.
   - Example: `HYPER_Project`
   
2. **Input Data:** Inside the project folder, create a subfolder named `input` and place the necessary input data files there.
   - Example: `HYPER_Project/input/b_observed.txt` (containing river flow records)

3. **Global Parameters:** Modify the `global_parameters.json` file to suit your specific project requirements.

### Installation

To install the HYPER package:

1. Navigate to the project directory in your command line interface.
2. Create a virtual environment:
   - Windows: `py -m venv .venv`
   - macOS / Linux: `python3 -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS / Linux: `source .venv/bin/activate`
4. Install the package:
   - `py -m pip install .`
   - For editable install: `py -m pip install -e .`

## Usage

Single Objective Optimization
To run a single objective optimization from the command line:
 `run-SO-opt`

Multi Objective Optimization
To run a multi-objective optimization from the command line:
./run-MO-opt.sh

If you are developing on the package you may wish to do an editable install: `py -m pip install -e .`


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The development of this toolbox was introduced in the paper by V. Yildiz and J. Vrugt, "A toolbox for the optimal design of run-of-river hydropower plants," published in Environmental Modelling &amp; Software.

