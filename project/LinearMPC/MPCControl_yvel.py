import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        """
        Build CVXPY problem for the y-velocity subsystem.
        The reduced model (self.A, self.B), steady-state (self.xs, self.us),
        horizon self.N and sampling self.Ts are available from the base class.
        """

        # sizes
        nx = self.nx
        nu = self.nu
        N = self.N

        #Tunable matrices
        Q = np.diag([1.0, 50.0, 200.0])  # wx, alpha, vy 
        R = np.diag([0.1])               # input d1

        #Terminal state computation
        _, P, _ = dlqr(self.A, self.B, Q, R)

        #Variable definition
        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        xref_param = cp.Parameter(nx, value=np.zeros(nx))
        uref_param = cp.Parameter(nu, value=np.zeros(nu))

        #Constraint definitin
        constraints = []

        # initial condition
        constraints += [x_var[:, 0] == x0_param]

        # dynamics constraints
        for k in range(N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]

        # state constraints: enforce tilt angle (alpha) within ±10°.
        alpha_idx = 1
        alpha_limit = np.deg2rad(10.0)
        xs_local = self.xs  # numeric array of length nx
        for k in range(N + 1):
            constraints += [
                xs_local[alpha_idx] + x_var[alpha_idx, k] <= alpha_limit, 
                xs_local[alpha_idx] + x_var[alpha_idx, k] >= -alpha_limit]

        # input constraints 
        d1_min = -np.deg2rad(15.0)
        d1_max = np.deg2rad(15.0)
        us_local = self.us  # numeric array of length nu
        for k in range(N):
            constraints += [
                us_local + u_var[:, k] <= d1_max,     # For d1: [-15°, +15°] per project
                us_local + u_var[:, k] >= d1_min]

        #Cost
        cost = 0
        for k in range(N):
            dx = x_var[:, k] - xref_param
            du = u_var[:, k] - uref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

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


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

        return u0, x_traj, u_traj
