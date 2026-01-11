import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base
from scipy.signal import place_poles

class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    #Tunable matrices
    Q = np.diag([20])  # vz 
    R = np.diag([0.1])  # input Pmean

    # state constraints: no limits for Vz
    state_constr_idx = 0
    state_constr_limit = np.inf
		
    # input constraints : min and max allowed power
    input_constr_min = 40.0
    input_constr_max = 80.0

    #Soft constraints
    use_soft_state_constraints = False
    use_soft_input_constraints = False
    Sx = 1e4   # state slack weight
    Su = 1e6   # input slack weight (if enabled)