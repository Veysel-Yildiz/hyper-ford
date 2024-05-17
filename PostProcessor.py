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
--------------------------------------
  Inputs:

    result : Optimization result
 """

# Import  the modules to be used from Library
import pandas as pd
 
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



    

