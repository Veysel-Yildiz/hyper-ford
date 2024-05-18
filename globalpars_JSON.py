"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""
""" global_parameters        
                v : the kinematics viscosity of water (m2/s)
                g : acceleration of gravity (m/s2)
               ng : generator-system efficiency
               hr : total hours in a year
                e : epsilon (m) for the relative roughness
               nf : specific spped range of francis turbine
               nk : specific spped range of kaplan turbine
               np : specific spped range of pelton turbine
               mf : min francis turbine design flow rate
               mk : min kaplan turbine design flow rate
               mp : min pelton turbine design flow rate
       eff_kaplan : Kaplan turbine efficiency
      eff_francis : Francis turbine efficiency
       eff_pelton : Pelton turbine efficiency
             perc : efficiency percentile.
             
                case_specific {These parameters can be changeg for each case study}
                hg: gross head(m)
                L : Penstock diameter (m)
               cf : site factor, used for the cost of the civil works
               om : maintenance and operation cost factor
              fxc : expropriation and other costs including transmission line
               ep : ectricity price in Turkey ($/kWh)
               pt : steel penstock price per ton ($/ton)
               ir : the investment discount rate (or interest rate)
                N : life time of the project (years)
 operating_scheme : turbine configuration setup 1 = 1 small + identical, 2 = all identical, 3 = all varied
       ObjectiveF : the objective function to be maximized  1: NPV, 2: BC             

"""


import json

# Define the global parameters
global_parameters = {
    "operating_scheme": 1,
    "ObjectiveF": 1,
    "case_specific": {
        "hg": 117.3,
        "L": 208,
        "cf": 0.15,
        "om": 0.01,
        "fxc": 5,
        "ep": 0.055,
        "pt": 1500,
        "ir": 0.095,
        "N": 49,
    },
    "e": 0.000045,
    "v": 1.004e-6,
    "g": 9.81,
    "ng": 0.98,
    "hr": 8760,
    "nf": [0.05, 0.33],
    "nk": [0.19, 1.55],
    "np": [0.005, 0.0612],
    "mf": 0.40,
    "mk": 0.20,
    "mp": 0.11,
    "eff_francis": [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0.0539,	0.1134,	0.1681,	0.2137,	0.2569,	0.2935,	0.3295,	0.361,	0.3904,	0.419,	0.4421,	0.4649,	0.4849,	0.5039,	0.521,	0.5374,	0.5518,	0.5684,	0.5833,	0.5967,	0.6093,	0.6225,	0.6345,	0.645,	0.6549,	0.6661,	0.676,	0.6859,	0.6967,	0.7062,	0.716,	0.7259,	0.7355,	0.7448,	0.7531,	0.7614,	0.77,	0.7775,	0.7848,	0.7928,	0.8007,	0.8082,	0.8156,	0.8227,	0.8294,	0.8356,	0.8418,	0.8479,	0.8539,	0.8596,	0.8653,	0.8708,	0.8761,	0.8807,	0.8851,	0.8886,	0.8921,	0.8953,	0.898,	0.9002,	0.9026,	0.9054,	0.9086,	0.9106,	0.9122,	0.9136,	0.9146,	0.915,	0.9144,	0.9135,	0.9118,	0.9093,	0.9071,	0.9033,	0.8981,	0.8938,	0.8886,	0.8827,	0.8754],
    "eff_pelton": [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0.005,	0.2875,	0.3965,	0.4822,	0.5592,	0.6222,	0.6687,	0.7027,	0.7307,	0.7532,	0.7694,	0.7842,	0.799,	0.8117,	0.8216,	0.8297,	0.8368,	0.8435,	0.8491,	0.854,	0.8603,	0.8652,	0.8699,	0.8746,	0.8793,	0.8834,	0.887,	0.8905,	0.8944,	0.8974,	0.9002,	0.9031,	0.9047,	0.9065,	0.9085,	0.9096,	0.9107,	0.9116,	0.9116,	0.9125,	0.9134,	0.914,	0.9142,	0.9138,	0.9138,	0.9154,	0.915,	0.9149,	0.9147,	0.9144,	0.9148,	0.9135,	0.914,	0.9141,	0.9136,	0.9129,	0.9124,	0.9122,	0.9122,	0.9116,	0.9106,	0.9098,	0.909,	0.9086,	0.9082,	0.9075,	0.9065,	0.9054,	0.9039,	0.9022,	0.9005,	0.8989,	0.8966,	0.8942,	0.8915,	0.8892,	0.8874,	0.8853,	0.883,	0.8805,	0.8782,	0.8765,	0.8734,	0.8702,	0.868,	0.8657,	0.8632,	0.8604,	0.8585,	0.8562],
    "eff_kaplan": [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0.25,	0.3116,	0.3786,	0.4353,	0.4807,	0.519,	0.5451,	0.5698,	0.5916,	0.6099,	0.6256,	0.6425,	0.6553,	0.6677,	0.6789,	0.6888,	0.6985,	0.7066,	0.7144,	0.7224,	0.7297,	0.7362,	0.7438,	0.7505,	0.7566,	0.7615,	0.7671,	0.7727,	0.7781,	0.7842,	0.79,	0.7953,	0.8004,	0.8059,	0.8116,	0.817,	0.8222,	0.8264,	0.8308,	0.8352,	0.8387,	0.8418,	0.8447,	0.8476,	0.8505,	0.8534,	0.8561,	0.859,	0.862,	0.864,	0.8664,	0.8691,	0.8721,	0.8751,	0.8776,	0.8797,	0.8816,	0.8838,	0.886,	0.8878,	0.8897,	0.8916,	0.8936,	0.8949,	0.896,	0.8967,	0.8971,	0.8971,	0.8964,	0.8963,	0.8951,	0.8938,	0.8931,	0.8923,	0.8915,	0.8907,	0.8898,	0.8884,	0.8876,	0.8872,	0.8869,	0.8849,	0.8825],
    "perc": [0,	0.01,	0.02, 0.03,	0.04,	0.05,	0.06,	0.07,	0.08,	0.09,	0.1,	0.11,	0.12,	0.13, 0.14, 0.15, 0.16,	0.17,	0.18,	0.19,	0.2,	0.21,	0.22,	0.23,	0.24,	0.25,	0.26,	0.27,	0.28,	0.29, 0.3,	0.31,	0.32,	0.33,	0.34,0.35,0.36,	0.37,	0.38,	0.39,	0.4, 0.41,	0.42,	0.43,	0.44,	0.45,	0.46,	0.47,	0.48,	0.49,	0.5,	0.51,	0.52,	0.53,	0.54,	0.55,	0.56,	0.57,	0.58,	0.59,	0.6,	0.61,	0.62, 0.63,	0.64,	0.65,	0.66,	0.67,	0.68,	0.69,	0.7,	0.71,	0.72,	0.73,	0.74,	0.75,	0.76,	0.77,	0.78,	0.79,	0.8,	0.81,	0.82,	0.83,	0.84,	0.85,	0.86,	0.87,	0.88,	0.89, 0.9,	0.91,	0.92,	0.93,	0.94,	0.95,	0.96,	0.97,	0.98,	0.99,	1],
}

# Save the parameters to a JSON file
with open('global_parameters.json', 'w') as json_file:
    json.dump(global_parameters, json_file, indent=4)

