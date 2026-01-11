import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    # Tunable matrices
    Q = np.diag([50.0, 50.0, 10.0, 20.0])  # [wy, beta, vx, x]
    R = np.diag([1.0])                      # [dP]

    # State constraints: beta within ±10°
    state_constr_idx = 1  # beta
    state_constr_limit = np.deg2rad(10.0)

    # Input constraints: ±15° flap angle
    input_constr_min = -np.deg2rad(15.0)
    input_constr_max = np.deg2rad(15.0)

    # Use soft constraints for nominal MPC
    use_soft_state_constraints = True
    use_soft_input_constraints = True
    Sx = 1e4
    Su = 1e4

    def _setup_controller(self) -> None:
        """Setup nominal MPC controller"""
        nx = self.nx
        nu = self.nu
        N = self.N

        # Compute LQR feedback gain K for terminal cost
        K, Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        self.Qf = Qf

        # Define CVXPY variables
        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        xref_param = cp.Parameter((nx,), name='xref', value=np.zeros(nx))
        uref_param = cp.Parameter((nu,), name='uref', value=np.zeros(nu))

        # Slack variables
        eps_x = cp.Variable((1, N + 1), nonneg=True, name="eps_x")
        eps_u = cp.Variable((1, N), nonneg=True, name="eps_u")

        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.xref_param = xref_param
        self.uref_param = uref_param
        self.eps_x = eps_x
        self.eps_u = eps_u

        # Cost function
        cost = 0
        for k in range(N):
            dx = x_var[:, k] - xref_param
            du = u_var[:, k] - uref_param
            cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)
            cost += self.Sx * eps_x[0, k]
            cost += self.Su * eps_u[0, k]
        
        # Terminal cost
        dxN = x_var[:, N] - xref_param
        cost += cp.quad_form(dxN, Qf)
        cost += self.Sx * eps_x[0, N]

        # Constraints
        constraints = []

        # Initial condition
        constraints.append(x_var[:, 0] == x0_param)

        # Dynamics
        for k in range(N):
            constraints.append(x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k])

        # Soft state constraints
        for k in range(N + 1):
            constraints.append(
                self.xs[self.state_constr_idx] + x_var[self.state_constr_idx, k] 
                <= self.state_constr_limit + eps_x[0, k]
            )
            constraints.append(
                self.xs[self.state_constr_idx] + x_var[self.state_constr_idx, k] 
                >= -self.state_constr_limit - eps_x[0, k]
            )

        # Soft input constraints
        for k in range(N):
            constraints.append(self.us + u_var[:, k] <= self.input_constr_max + eps_u[0, k])
            constraints.append(self.us + u_var[:, k] >= self.input_constr_min - eps_u[0, k])

        # Define optimization problem
        objective = cp.Minimize(cost)
        self.ocp = cp.Problem(objective, constraints)
        
        print(f"Nominal MPC for X: Setup complete")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve nominal MPC and return control input"""
        
        # Default targets
        if x_target is None:
            x_target = self.xs.copy()
        if u_target is None:
            u_target = self.us.copy()
        
        # Work in deviation coordinates
        x0_dev = x0 - self.xs
        xref_dev = x_target - self.xs
        uref_dev = u_target - self.us
        
        # Set parameters
        self.x0_param.value = x0_dev
        self.xref_param.value = xref_dev
        self.uref_param.value = uref_dev
        
        # Solve
        self.ocp.solve(solver=cp.PIQP, verbose=False)
        
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Nominal MPC X: Solver status: {self.ocp.status}")
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj
        
        # Get solution
        x_opt_dev = np.array(self.x_var.value)
        u_opt_dev = np.array(self.u_var.value)
        
        # Convert back to absolute coordinates
        u0 = u_opt_dev[:, 0] + self.us
        x_traj = x_opt_dev + self.xs.reshape(-1, 1)
        u_traj = u_opt_dev + self.us.reshape(-1, 1)
        
        return u0.flatten(), x_traj, u_traj