import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])
 
    #Tunable matrices
    Q = np.diag([50.0, 10.0])  # wz, gamma 
    R = np.diag([1.])        # input Pdiff
 
    # state constraints: enforce roll angle (gamma) within ±10°.
    state_constr_idx = 1
    state_constr_limit = np.deg2rad(40.0)
 
	# input constraints  For Pdiff: [-20%, +20%] 
    input_constr_min = -20.0
    input_constr_max =  20.0

    #Soft constraints
    use_soft_state_constraints = False
    use_soft_input_constraints = False
    Sx = 1e4   # state slack weight
    Su = 1e6   # input slack weight (if enabled)


