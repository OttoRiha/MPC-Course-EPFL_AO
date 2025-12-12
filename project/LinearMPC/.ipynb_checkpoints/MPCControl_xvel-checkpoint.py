import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base): #WHY IS X NOT A STATE? IT IS IN THE LINEARISED MODEL
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1]),2

    def _setup_controller(self) -> None:
        """
        Build CVXPY problem for the x-velocity subsystem.
        The reduced model (self.A, self.B), steady-state (self.xs, self.us),
        horizon self.N and sampling self.Ts are available from the base class.
        """

        # sizes
        nx = self.nx
        nu = self.nu
        N = self.N

        Q = np.diag([1.0, 50.0, 200.0])  # wy, beta, vx 
        R = np.diag([0.1])               # input d2

        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        # ---------------------------------------------------------------------
        # 3) Constraints (deviation variables)
        #    We form constraints on absolute variables where useful:
        #      x_abs = xs + x_dev
        #      u_abs = us + u_dev
        # ---------------------------------------------------------------------
        constraints = []

        # initial condition
        constraints += [x[:, 0] == x0_param]

        # dynamics constraints
        for k in range(N):
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

        # state constraints: enforce tilt angle (beta) within ±10°.
        # In this reduced ordering beta is the second state -> index 1
        beta_idx = 1
        beta_limit = np.deg2rad(10.0)
        # absolute beta = xs[beta_local] + x[beta_local,k]
        xs_local = self.xs  # numeric array of length nx
        for k in range(N + 1):
            constraints += [
                xs_local[beta_idx] + x[beta_idx, k] <= beta_limit,
                xs_local[beta_idx] + x[beta_idx, k] >= -beta_limit,
            ]

        # input constraints (absolute bounds)
        # For d2: [-15°, +15°] per project
        d2_min = -np.deg2rad(15.0)
        d2_max = np.deg2rad(15.0)
        us_local = self.us  # numeric array of length nu
        for k in range(N):
            # absolute input = us_local + u[:,k]
            constraints += [
                us_local + u[:, k] <= d2_max,
                us_local + u[:, k] >= d2_min,
            ]

        # (optionally) you could also constrain rate-of-change of u by adding:
        # for k in range(N-1):
        #     constraints += [cp.abs(u[:,k+1] - u[:,k]) <= delta_u_max]

        # ---------------------------------------------------------------------
        # 4) Cost
        # ---------------------------------------------------------------------
        cost = 0
        for k in range(N):
            dx = x[:, k] - xref_param
            du = u[:, k] - uref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        # terminal cost
        dxN = x[:, N] - xref_param
        cost += cp.quad_form(dxN, P)

        objective = cp.Minimize(cost)

        # ---------------------------------------------------------------------
        # 5) Build problem and store handles
        # ---------------------------------------------------------------------
        self.x_var = x
        self.u_var = u
        self.x0_param = x0_param
        self.xref_param = xref_param
        self.uref_param = uref_param

        # Build problem
        self.ocp = cp.Problem(objective, constraints)

        # solver options as attributes for later use
        self.solver_opts = {"verbose": False, "warm_start": True}

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