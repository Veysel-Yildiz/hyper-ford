"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" Return :
op_table: optimization table constructed with the optimization result parameters,
 including the objective function value, turbine type, turbine configuration, 
 penstock diameter, and turbine design discharges. 
 
Scatter Plot: Pareto Front of design alternatives 
--------------------------------------
  Inputs:

    result : Optimization result
 """

# Import  the modules to be used from Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
def postplot(result): 

 # Extract design parameters
 OF = abs(result['fun'])  # Objective Function value
 typet = round(result['x'][0])  # Turbine type
 conf = round(result['x'][1])  # Turbine configuration
 diameter = result['x'][2]  # Diameter
 design_discharges = result['x'][3:]  # Design discharges

 # Map typet to turbine type name
 turbine_type_map = {1: "Kaplan", 2: "Francis", 3: "Pelton"}
 turbine_type = turbine_type_map.get(typet, "Unknown")

 # Map conf to turbine configuration name
 if conf == 1:
    turbine_config = "single"
 elif conf == 2:
    turbine_config = "dual"
 elif conf == 3:
    turbine_config = "triple"
 else:
    turbine_config = f"{conf}th"

 # Create a dictionary for the table
 data = {
    'OF': [OF],
    'Turbine Type': [turbine_type],
    'Turbine Config': [turbine_config],
    'Diameter (m)': [diameter]
 }

 # Add design discharges to the dictionary
 for i, discharge in enumerate(design_discharges, start=1):
     data[f'Design Discharge {i} m3/s'] = [discharge]

 # Convert dictionary to DataFrame
 op_table = pd.DataFrame(data)

 return op_table


def MO_postplot(F_opt, X_opt):

    # Extract design parameters
    NPV = abs(F_opt[:, 0])  # Objective Function value
    BC = abs(F_opt[:, 1])  # Objective Function value

    typet = np.round(X_opt[:, 0]).astype(int)  # Turbine type
    conf = np.round(X_opt[:, 1]).astype(int)  # Turbine type
    
    diameter = X_opt[:, 2]  # Diameter
    design_discharges = X_opt[:, 3:]  # Design discharges

    # Map typet to turbine type name
    turbine_type_map = {1: "Kaplan", 2: "Francis", 3: "Pelton"}
    turbine_type = [turbine_type_map.get(t, "Unknown") for t in typet]

    # Map conf to turbine configuration name
    turbine_config_map = {1: "single", 2: "dual", 3: "triple"}
    turbine_config = [turbine_config_map.get(c, f"{c}th") for c in conf]

    # Create a dictionary for the table
    data = {
        'NPV': NPV,
        'BC': BC,
        'Turbine Type': turbine_type,
        'Turbine Config': turbine_config,
        'Diameter (m)': diameter
    }

    # Add design discharges to the dictionary
    for i, discharge in enumerate(design_discharges.T, start=1):
        data[f'Design Discharge {i} m3/s'] = discharge

    # Convert dictionary to DataFrame
    op_table = pd.DataFrame(data)

    return op_table
    

def MO_scatterplot(F1, F2):

    plt.scatter(-1 * F1, -1 * F2, color='blue')
    plt.xlabel("NPV (Million USD)", fontsize=15)
    plt.ylabel("BC (-)", fontsize=16)
    plt.title("Optimization Results", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    plt.show()






