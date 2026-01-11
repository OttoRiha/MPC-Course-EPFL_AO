import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base): #WHY IS X NOT A STATE? IT IS IN THE LINEARISED MODEL
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1]) #,2 #what is that two here?
	
	#Tunable matrices
    Q = np.diag([10.0, 10.0, 10.0])  # wy, beta, vx 
    R = np.diag([1.])               # input d2
	
	# state constraints: enforce tilt angle (beta) within ±10°.
    state_constr_idx = 1
    state_constr_limit = np.deg2rad(10.0)
	
    # input constraints : max 15° flap angle(delta2)
    input_constr_min = -np.deg2rad(15.0)
    input_constr_max = np.deg2rad(15.0)	

    #Soft constraints
    use_soft_state_constraints = True
    use_soft_input_constraints = False
    Sx = 1e4   # state slack weight
    Su = 1e6   # input slack weight (if enabled)

   