import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    # Tunable matrices
    Q = np.diag([50.0, 50.0, 10.0, 20.0])  # [wx, alpha, vy, y]
    R = np.diag([1.0])                     # input d1
 
    # state constraints: enforce tilt angle (alpha) within ±10°.
    state_constr_idx = 1
    state_constr_limit = np.deg2rad(10.0)
 
    # input constraints : max 15° flap angle (delta1)
    input_constr_min = -np.deg2rad(15.0)
    input_constr_max = np.deg2rad(15.0)

    #Soft constraints
    use_soft_state_constraints = True
    use_soft_input_constraints = False
    Sx = 1e4   # state slack weight
    Su = 1e6   # input slack weight (if enabled)
