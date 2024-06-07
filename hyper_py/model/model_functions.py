## HYPER SIMULATION & OPTIMISATION
"""
############################################################################
#                     Written by Veysel Yildiz                             #
#                      vyildiz1@sheffield.ac.uk                            #
#                   The University of Sheffield,June 2024                  #
############################################################################
"""

# Import  the modules to be used from Library
import numpy as np
from itertools import combinations


##################################################COST#########################


def cost(design_ic, design_h, typet, conf, D, global_parameters):
    """Return:
         cost_em : Electro-mechanic (turbine) cost in million USD
        cost_pen : Penstock cost in million USD
         cost_ph : Powerhouse cost in million USD
    --------------------------------------
       Inputs :

            HP : structure with variables used in calculation
     design_ic : installed capacity
      design_h : design head
         typet : turbine type
          conf : turbine configuration; single, dual, triple
             D : penstock diameter
            pt : steel penstock price per ton ($)
    """
    case_specific = global_parameters["case_specific"]
    pt = case_specific["pt"]
    L = case_specific["L"]

    tp = 8.4 / 1000 * D + 0.002  # Thickness of the pipe  [m]

    # tp1 = (1.2 * HP.hd * D / ( 20* 1.1) +2)*0.1; #t = (head + water hammer head (20%))*D / The working stress of the steel * 2
    # tp2 = (D + 0.8) / 4; % min thickness in cm
    # tp = max(tp1, tp2)/100;% min thickness in m

    cost_pen = np.pi * tp * D * L * 7.874 * pt / 10**6

    # Calculate the cost of power house (in M dollars)
    cost_ph = 200 * (design_ic / 1000) ** -0.301 * design_ic / 10**6

    # Calculate the cost of power house (in Mdollars)
    # cost_ph = HP.power*100/10**6;
    # cost_ph = HP.power / 10000;

    # Switch among the different turbine combinations

    if typet == 2:  # Francis turbine cost
        cost_em = (
            2.927
            * (design_ic / 1000) ** 1.174
            * (design_h) ** -0.4933
            * 1.1
            * (1 + (conf - 1) * (conf - 2) * 0.03)
        )  # in $

    elif typet == 3:  # pelton turbine cost
        cost_em = (
            1.984
            * (design_ic / 1000) ** 1.427
            * (design_h) ** -0.4808
            * 1.1
            * (1 + (conf - 1) * (conf - 2) * 0.03)
        )  # in $

    else:  # Kaplan turbine cost

        cost_em = (
            2.76
            * (design_ic / 1000) ** 0.5774
            * (design_h) ** -0.1193
            * 1.1
            * (1 + (conf - 1) * (conf - 2) * 0.03)
        )  # in $

    # return cost_em , cost_pen,  cost_ph #tp,
    return np.float64(cost_em), np.float64(cost_pen), np.float64(cost_ph)


################################################## MOODY ########################


def moody(ed, Re):
    """Return f, friction factor
    --------------------------------------
      Inputs:

        HP : structure with variables used in calculation
        ed : the relative roughness: epsilon / diameter.
        Re : the Reynolds number

    """
    f = np.zeros_like(Re)

    # Find the indices for Laminar, Transitional and Turbulent flow regimes

    LamR = np.where((0 < Re) & (Re < 2000))
    LamT = np.where(Re > 4000)
    LamTrans = np.where((2000 < Re) & (Re < 4000))

    f[LamR] = 64 / Re[LamR]

    # Calculate friction factor for Turbulent flow using the Colebrook-White approximation
    f[LamT] = 1.325 / (np.log(ed / 3.7 + 5.74 / (Re[LamT] ** 0.9)) ** 2)

    Y3 = -0.86859 * np.log(ed / 3.7 + 5.74 / (4000**0.9))
    Y2 = ed / 3.7 + 5.74 / (Re[LamTrans] ** 0.9)
    FA = Y3 ** (-2)
    FB = FA * (2 - 0.00514215 / (Y2 * Y3))
    R = Re[LamTrans] / 2000
    X1 = 7 * FA - FB
    X2 = 0.128 - 17 * FA + 2.5 * FB
    X3 = -0.128 + 13 * FA - 2 * FB
    X4 = R * (0.032 - 3 * FA + 0.5 * FB)
    f[LamTrans] = X1 + R * (X2 + R * (X3 + X4))

    return f


################################################## operation optimization  #########################


# @jit(nopython=True)
def inflow_allocation(nr, Od, q_inc, kmin, perc, func_Eff):
    """Return:

              qt : Turbine inflow for each incremental step.
          Eff_qi : Efficiency and inflow multiplication for energy calculation.
             nrc : Turbine running capacity as a ratio
    --------------------------------------
        Inputs:

              nr : Turbine random sampled allocated discharge.
              Od : Turbine design discharge.
           q_inc : Incremental flow steps between turbine min and  max (design) discharge.
            kmin : Minimum turbine discharge to operate.
            perc : Efficiency percentile.
        func_Eff : Efficiency curve.

    """

    # Multiply each row of nr by the corresponding element of q_inc
    nrc = nr * q_inc

    # Calculate qt as the minimum of nrc and Od
    qt = np.minimum(nrc, Od)

    # Interpolate values from func_Eff based on qt/Od ratio
    Daily_Efficiency = np.interp(qt / Od, perc, func_Eff)

    # Set qt and nrc to zero where qt is less than kmin * Od
    idx = qt < kmin * Od
    qt[idx] = 0
    nrc[idx] = 0

    # Calculate np as the product of Efficiency and qt
    Eff_qi = Daily_Efficiency * qt

    return qt, Eff_qi, nrc


################################################## possible combinations #########################


# @jit(nopython=True)
def generate_patterns(maxturbine):
    # Function to generate the required combinations

    """Return pattern, all possible combinations of turbines at full capacity
    --------------------------------------
    Inputs:

    maxturbine : Number of turbine

    """

    patterns = []  # Initialize an empty list to store patterns

    # Generate all possible patterns of 1s and 0s for the given maxturbine,
    for num_ones in range(1, maxturbine + 1):

        # Iterate over combinations of indices for placing 1s
        for comb in combinations(range(maxturbine), num_ones):

            pattern = [0] * maxturbine  # Initialize pattern with all 0s
            for index in comb:  # Set indices corresponding to 1s
                pattern[index] = 1
            patterns.append(pattern)  # Add the pattern to the list
    return patterns


################################################## daily power #############################


def operation_optimization(
    Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters
):
    """Return dailyPower, daily generated power output based on operation optimization
    --------------------------------------
      Inputs:
               Q : daily discharge
      maxturbine : Number of turbines
        Qturbine : Turbine's design discharge
               D : Penstock diameter
            kmin : Technical min flow rate of turbine to operate
        func_Eff : Efficiency curve
              hg : gross head(m)
               L : Penstock diameter (m)
            perc : efficiency percentile.
               e : epsilon (m) for the relative roughness

    """
    ## unpack global variables
    e = global_parameters["e"]
    case_specific = global_parameters["case_specific"]
    L = case_specific["L"]
    hg = case_specific["hg"]
    perc = global_parameters["perc"]

    maxT = len(Q)  # the size of time steps

    Ns = 1000  # size of the random sample

    # Define the number of rows for discretization
    rowCount = 1000

    # Find the minimum value in the array and multiply kmin
    minflow = kmin * np.min(Qturbine)

    # Create an array 'q_inc' using linspace with 'rowCount' elements
    # 'minflow' is the starting value, 'Q_design' is the ending value,
    # and 'rowCount' is the number of elements to generate
    q_inc = np.linspace(minflow, Q_design, rowCount)

    # Generate all random values at once
    nr = np.random.rand(Ns, maxturbine, rowCount)

    # Generate patterns for the current maxturbine value
    # This is to make sure that turbines will be sampled at full capacity
    patterns = generate_patterns(maxturbine)

    # Apply the generated patterns to the nr array
    for i, pattern in enumerate(patterns):
        if i >= Ns:  # Avoid going out of bounds
            break
        nr[i, :, :] = np.array(pattern)[:, np.newaxis]

    # Normalize so the sum is 1 along the second dimension (axis=1)
    # This is equivalent to dividing each row of 'nr' by the sum of the corresponding row
    nr = nr / np.sum(nr, axis=1, keepdims=True)

    # Create arrays filled with zeros
    q = np.zeros((Ns, rowCount))
    Eff_q = np.zeros((Ns, rowCount))

    # Loop through each value of On
    for i in range(maxturbine):
        # Perform Voperation_OPT operation
        qi, Eff_qi, _ = inflow_allocation(
            nr[:, i, :], Qturbine[i], q_inc, kmin, perc, func_Eff
        )

        # Update q and nP arrays
        q += qi
        Eff_q += Eff_qi

        # Calculate flow velocity in the pipe
        V = 4 * q / (np.pi * D**2)

        # Calculate the Reynolds number
        Re = V * D / 10**-6  # kinematic viscosity ν = 1,002 · 10−6 m2∕s

        ed = e / D  # calculate the relative roughness: epsilon / diameter.

        # Find f, the friction factor [-]
        f = moody(ed, Re)

        # Calculate the head loss due to friction in the penstock
        hnet = hg - f * (L / D) * V**2 / 19.62 * 1.1

        # Calculate DP
        DP = Eff_q * hnet * 9.6138  # DP = 9.81 * ng;

        # Find the index of the maximum value in each column of DP
        id = np.argmax(DP, axis=0)

        # Create Ptable
        Ptable = np.column_stack((q_inc, DP[id, np.arange(rowCount)]))

        ## Initialize operating_mode array with NaN values
        # operating_mode = np.full((rowCount, On), np.nan) # allocated discharge
        ## Loop through each row
        # for i in range(rowCount):
        ## Copy values from nr[id[i], :, i] to operating_mode[i, :]
        # operating_mode[i, :] = nr[id[i], :, i]

        # Extract TableFlow and TablePower
        TableFlow = Ptable[:, 0]
        TablePower = Ptable[:, 1]

        # Pre-allocate output variable
        dailyPower = np.zeros(maxT)

        # Calculate sum of Od1 and Od2
        qw = np.minimum(Q, Q_design)

        # Find the indices corresponding to qw < minflow
        # shutDownIndices = np.where(qw < minflow)[0]

        # Find the indices corresponding to qw >= minflow
        activeIndices = np.where(qw >= minflow)[0]

        # Calculate pairwise distances between qw(activeIndices) and TableFlow
        distances = np.abs(qw[activeIndices][:, np.newaxis] - TableFlow[np.newaxis, :])

        # Find the indices of TableFlow closest to qw for active turbines
        indices = np.argmin(distances, axis=1)

        # Assign TablePower values to active turbines based on the indices
        dailyPower[activeIndices] = TablePower[indices]

    return dailyPower


################################################## Sampled daily power #############


def S_operation_optimization(
    Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters
):
    """Return dailyPower, daily generated power output based on operation optimization
    --------------------------------------
      Inputs:
               Q : daily discharge
      maxturbine : Number of turbines
        Qturbine : Turbine's design discharge
               D : Penstock diameter
            kmin : Technical min flow rate of turbine to operate
        func_Eff : Efficiency curve
              hg : gross head(m)
               L : Penstock diameter (m)
            perc : efficiency percentile.
               e : epsilon (m) for the relative roughness

    """
    ## unpack global variables
    e = global_parameters["e"]
    case_specific = global_parameters["case_specific"]
    L = case_specific["L"]
    hg = case_specific["hg"]
    perc = global_parameters["perc"]

    maxT = len(Q)  # the size of time steps

    Ns = 1000  # size of the random sample

    # Define the number of rows for discretization
    rowCount = maxT

    # Generate all random values at once
    nr = np.random.rand(Ns, maxturbine, rowCount)

    # Generate patterns for the current maxturbine value
    # This is to make sure that turbines will be sampled at full capacity
    patterns = generate_patterns(maxturbine)

    # Apply the generated patterns to the nr array
    for i, pattern in enumerate(patterns):
        if i >= Ns:  # Avoid going out of bounds
            break
        nr[i, :, :] = np.array(pattern)[:, np.newaxis]

    # Normalize so the sum is 1 along the second dimension (axis=1)
    # This is equivalent to dividing each row of 'nr' by the sum of the corresponding row
    nr = nr / np.sum(nr, axis=1, keepdims=True)

    # Create arrays filled with zeros
    q = np.zeros((Ns, rowCount))
    Eff_q = np.zeros((Ns, rowCount))

    # Loop through each value of On
    for i in range(maxturbine):
        # Perform Voperation_OPT operation
        qi, Eff_qi, _ = inflow_allocation(
            nr[:, i, :], Qturbine[i], Q, kmin, perc, func_Eff
        )

        # Update q and nP arrays
        q += qi
        Eff_q += Eff_qi

        # Calculate flow velocity in the pipe
        V = 4 * q / (np.pi * D**2)

        # Calculate the Reynolds number
        Re = V * D / 10**-6  # kinematic viscosity ν = 1,002 · 10−6 m2∕s

        ed = e / D  # calculate the relative roughness: epsilon / diameter.

        # Find f, the friction factor [-]
        f = moody(ed, Re)

        # Calculate the head loss due to friction in the penstock
        hnet = hg - f * (L / D) * V**2 / 19.62 * 1.1

        # Calculate DP
        DP = Eff_q * hnet * 9.6138  # DP = 9.81 * ng;

        # Find the index of the maximum value in each column of DP
        id = np.argmax(DP, axis=0)

        # Create Ptable
        Ptable = np.column_stack((Q, DP[id, np.arange(rowCount)]))

        # Extract TableFlow and TablePower
        # TableFlow = Ptable[:, 0]
        dailyPower = Ptable[:, 1]

    return dailyPower


################################################## possible combinations #########################


def get_sampled_data(streamflow, sample_size):
    """Return sampled data, daily discharge values
    --------------------------------------
    Inputs:
     streamflow : record of streamflow values
     sample_size : sample size of the data

    """
    # Sort the array in descending order
    sorted_array = np.sort(streamflow)[::-1]

    # Ensure sample_size is valid
    if sample_size < 2:
        raise ValueError("Sample size should be at least 2")

    # Ensure sample_size is not greater than array length
    if sample_size > len(sorted_array):
        raise ValueError("Sample size should not exceed the length of the array")

    # Get the index of the maximum and minimum values
    # max_index = 0
    # min_index = -1

    # Get indices for the rest of the sample
    sample_indices = np.linspace(1, len(sorted_array) - 2, sample_size - 2, dtype=int)

    # Concatenate max, sample, and min indices
    indices = np.concatenate(([0], sample_indices, [-1]))

    # Return subset of the sorted array based on the selected indices
    return sorted_array[indices]
