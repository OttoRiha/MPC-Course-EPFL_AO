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
    R = np.diag([1.0])                      # [dR]

    # State constraints: alpha within ±10°
    state_constr_idx = 1  # alpha
    state_constr_limit = np.deg2rad(10.0)

    # Input constraints: ±15° flap angle
    input_constr_min = -np.deg2rad(15.0)
    input_constr_max = np.deg2rad(15.0)

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        self.ocp = ...

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        u0 = ...
        x_traj = ...
        u_traj = ...

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj