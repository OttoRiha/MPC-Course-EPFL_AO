import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])
 
    #Tunable matrices
    Q: np.ndarray = np.diag([1.0, 50.0, 200.0])  # wx, alpha, vy 
    R = np.diag([0.1])               # input d1
 
    # state constraints: enforce tilt angle (alpha) within ±10°.
    state_constr_idx = 1
    state_constr_limit = np.deg2rad(10.0)
 
    # input constraints : max 15° flap angle (delta1)
    input_constr_min = -np.deg2rad(15.0)
    input_constr_max = np.deg2rad(15.0)

