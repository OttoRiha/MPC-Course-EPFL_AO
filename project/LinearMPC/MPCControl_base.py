import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # sizes
        nx = self.nx
        nu = self.nu
        N = self.N

        #Tunable matrices
        # Q = np.diag([50.0, 50.0])  # wz, gamma 
        # R = np.diag([0.1])        # input Pdiff

        #Terminal state computation
        _, P, _ = dlqr(self.A, self.B, self.Q, self.R)

        #Variable definition
        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        xref_param = cp.Parameter(nx, value=np.zeros(nx))
        uref_param = cp.Parameter(nu, value=np.zeros(nu))

        #Constraint definition
        constraints = []

        # initial condition
        constraints += [x_var[:, 0] == x0_param]

        # dynamics constraints
        for k in range(N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]

        # state constraints: enforce tilt angle (gamma) within ±10°.
        #state_constr_idx = 1
        #state_constr_limit = np.deg2rad(10.0)
        xs_local = self.xs  # numeric array of length nx
        for k in range(N + 1):
            constraints += [
                xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k] <= self.state_constr_limit, 
                xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k] >= -self.state_constr_limit]

        # input constraints  For Pdiff: [-20%, +20%] 
        #input_constr_min = -20.0
        #input_constr_max =  20.0
        us_local = self.us  # numeric array of length nu
        for k in range(N):
            constraints += [
                us_local + u_var[:, k] <= self.input_constr_max,
                us_local + u_var[:, k] >= self.input_constr_min]

        #Cost
        cost = 0
        for k in range(N):
            dx = x_var[:, k] - xref_param
            du = u_var[:, k] - uref_param
            cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)

        # terminal cost
        dxN = x_var[:, N] - xref_param
        cost += cp.quad_form(dxN, P)

        #Problem
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.xref_param = xref_param
        self.uref_param = uref_param

        # Build problem
        objective = cp.Minimize(cost)
        self.ocp = cp.Problem(objective, constraints)

        # solver options as attributes for later use
        self.solver_opts = {"verbose": False, "warm_start": True}

        # YOUR CODE HERE
        #################################################

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # Default targets are steady-state (absolute)
        if x_target is None:
            x_target = self.xs.copy()
        if u_target is None:
            u_target = self.us.copy()

        # Work in deviation coordinates for the solver:
        x0_dev = x0 - self.xs
        xref_dev = x_target - self.xs
        uref_dev = u_target - self.us

        # set CVXPY parameter values
        self.x0_param.value = x0_dev
        self.xref_param.value = xref_dev
        self.uref_param.value = uref_dev

        #Solve
        self.ocp.solve(solver=cp.OSQP, **self.solver_opts)

        #retrieve results and convert back to absolute coordinates
        x_opt_dev = np.array(self.x_var.value)
        u_opt_dev = np.array(self.u_var.value)

        #Returns without deviation
        x_traj = x_opt_dev + self.xs.reshape(-1, 1)
        u_traj = u_opt_dev + self.us.reshape(-1, 1)

        # first input to apply (absolute)
        u0 = u_traj[:, 0].flatten()
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
