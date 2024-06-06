"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""

"""
This script provides functions to handle parameter validation for engineering applications.

1. get_parameter_constraints():
   - Retrieves parameter constraints from a JSON file.
   - Loads the constraints into a dictionary for validation.

2. validate_input(name, value, constraint):
   - Validates a single parameter input to ensure it meets the defined constraints.
   - Compares the input value against the specified constraints for the parameter.
   - Returns True if the input value meets the constraints, False otherwise.

3. validate_parameters(parameters, constraints):
   - Validates multiple parameters to ensure they meet the defined constraints.
   - Iterates through the dictionary of parameters and their values.
   - Uses validate_input() function to check each parameter against its constraints.
   - Returns True if all parameters meet their respective constraints, False otherwise.

Usage:
- Load parameter constraints using get_parameter_constraints() function.
- Validate individual parameters using validate_input() function.
- Validate sets of parameters using validate_parameters() function.

"""


import json
import re

# Load the parameters and constraints from the JSON file
with open('global_parameters.json', 'r') as json_file:
    global_parameters = json.load(json_file)

def get_parameter_constraints():

    """
    Retrieves the parameter constraints from a JSON file.

    Returns:
    dict: Dictionary containing parameter constraints.
   """

    return {
        "operating_scheme": {"type": "int", "min": 1, "max": 3, "unit": ""},
        "ObjectiveF": {"type": "int", "min": 1, "max": 2, "unit": ""},
        "case_specific": {
            "hg": {"type": "float", "min": 0, "max": 1000, "unit": "meters"},
            "L": {"type": "float", "min": 0, "max": 10000, "unit": "meters"},
            "cf": {"type": "float", "min": 0.0, "max": 1.0, "unit": ""},
            "om": {"type": "float", "min": 0.0, "max": 0.5, "unit": ""},
            "fxc": {"type": "int", "min": 0, "max": 1000, "unit": "millions USD"},
            "ep": {"type": "float", "min": 0.0, "max": 0.5, "unit": "USD/kWh"},
            "pt": {"type": "int", "min": 0, "max": 15000, "unit": "USD/tones"},
            "ir": {"type": "float", "min": 0.0, "max": 0.5, "unit": ""},
            "N": {"type": "int", "min": 1, "max": 100, "unit": "years"},
        },
        "e": {"type": "float", "min": 0.0, "max": 0.0001, "unit": ""},
        "v": {"type": "float", "min": 1.0e-7, "max": 1.0e-5, "unit": ""},
        "g": {"type": "float", "min": 9.0, "max": 10.0, "unit": "m/s²"},
        "ng": {"type": "float", "min": 0.8, "max": 1.0, "unit": ""},
        "hr": {"type": "int", "min": 1, "max": 8760, "unit": "hours"},
        "nf": {"type": "list", "min": 0.0, "max": 1.0, "unit": ""},
        "nk": {"type": "list", "min": 0.0, "max": 2.0, "unit": ""},
        "np": {"type": "list", "min": 0.0, "max": 0.1, "unit": ""},
        "mf": {"type": "float", "min": 0.0, "max": 0.5, "unit": ""},
        "mk": {"type": "float", "min": 0.0, "max": 0.5, "unit": ""},
        "mp": {"type": "float", "min": 0.0, "max": 0.5, "unit": ""},
        "eff_francis": {"type": "list", "min": 0.0, "max": 1.0, "unit": ""},
        "eff_pelton": {"type": "list", "min": 0.0, "max": 1.0, "unit": ""},
        "eff_kaplan": {"type": "list", "min": 0.0, "max": 1.0, "unit": ""},
        "perc": {"type": "list", "min": 0.0, "max": 1.0, "unit": ""},
    }

def validate_input(name, value, constraint):
    """
    Validates a single parameter input to ensure it meets the defined constraints.
    
    Parameters:
        name (str): The name of the parameter being validated.
        value (any): The value of the parameter to be validated.
        constraint (dict): Dictionary containing constraints for the parameter.
    
    Returns:
        bool: True if the input value meets the constraints, False otherwise.
    """
    value_type = constraint.get('type')
    min_value = constraint.get('min')
    max_value = constraint.get('max')
    regex = constraint.get('regex')
    unit = constraint.get('unit', '')

    # Type checking
    if value_type == 'int' and not isinstance(value, int):
        raise ValueError(f"{name} should be an integer, but got {type(value).__name__}")
    elif value_type == 'float' and not isinstance(value, (float, int)):
        raise ValueError(f"{name} should be a float or integer, but got {type(value).__name__}")
    elif value_type == 'string' and not isinstance(value, str):
        raise ValueError(f"{name} should be a string, but got {type(value).__name__}")
    elif value_type == 'list' and not isinstance(value, list):
        raise ValueError(f"{name} should be a list, but got {type(value).__name__}")

    # Range checking for numerical types
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} should be at least {min_value} {unit}, but got {value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} should be at most {max_value} {unit}, but got {value}")
    # Range checking for lists
    elif isinstance(value, list):
        for item in value:
            if min_value is not None and item < min_value:
                raise ValueError(f"All elements in {name} should be at least {min_value} {unit}, but got {item}")
            if max_value is not None and item > max_value:
                raise ValueError(f"All elements in {name} should be at most {max_value} {unit}, but got {item}")

    # Regex checking
    if regex is not None and not re.match(regex, value):
        raise ValueError(f"{name} should match the pattern {regex}")

def validate_parameters(parameters, constraints):
    """
    Validates multiple parameters to ensure they meet the defined constraints.
    
    Parameters:
        parameters (dict): Dictionary containing parameter names and their values.
        constraints (dict): Dictionary containing constraints for parameter validation.
    
    Returns:
        bool: True if all parameters meet their respective constraints, False otherwise.
    """
    for category, params in parameters.items():
        if isinstance(params, dict):
            for param, value in params.items():
                if param in constraints.get(category, {}):
                    validate_input(param, value, constraints[category][param])
        else:
            if category in constraints:
                validate_input(category, params, constraints[category])
