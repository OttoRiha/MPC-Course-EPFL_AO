import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt


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



    # Compute minimal robust invariant set
    def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep()  # optionally: Omega_next.minVrep()
            if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 1e-2:
                print('Minimal robust invariant set computation converged after {0} iterations.'.format(itr))
                break

            if itr == max_iter:
                print('Minimal robust invariant set computation did NOT converge after {0} iterations.'.format(itr))
            
            Omega = Omega_next
            itr += 1
        return Omega_next




    def max_invariant_set(A_cl, X: Polyhedron, mpc_type: str, max_iter=100) -> Polyhedron:
        """
        Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
        """
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)),np.vstack((f, f)).reshape((-1,)))
            O.minHrep(True)
            # Temporary fix since contains() is not robust enough
            #_ = O.Vrep
            if O == Oprev:
                converged = True
                break
            itr += 1
        if converged:
            print(f"Maximum invariant set successfully computed after {itr} iterations " f"for {mpc_type} MPC.")

            #print('Maximum invariant set successfully computed after {0} iterations'.format(itr))
            # --- Added debug prints ---
            # print("X.A shape:", X.A.shape)
            # print("X.b shape:", X.b.shape)
            # print("C_inf A shape:", None if O.A is None else O.A.shape)
            # print("C_inf b shape:", None if O.b is None else O.b.shape)
        else:
            print(f"Not converged"f"for {mpc_type} MPC.")

        return O


    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # sizes
        nx = self.nx
        nu = self.nu
        N = self.N

        #Terminal state computation
        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        K=-K

        #Variable definition
        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        xref_param = cp.Parameter(nx, value=np.zeros(nx))
        uref_param = cp.Parameter(nu, value=np.zeros(nu))

        # Slack variables definition
        if self.use_soft_state_constraints:
            eps_x = cp.Variable((1, N + 1), nonneg=True, name="eps_x")

        if self.use_soft_input_constraints:
            eps_u = cp.Variable((1, N), nonneg=True, name="eps_u")

        #Constraint definition
        constraints = []

        # initial condition
        constraints += [x_var[:, 0] == x0_param]

        # dynamics constraints
        for k in range(N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]

        # state constraints: depends on the self.state_constr_idx and self.state_constr_limit
        xs_local = self.xs  # numeric array of length nx
        for k in range(N + 1):
            if self.use_soft_state_constraints:
                constraints += [
                    xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k]<= self.state_constr_limit + eps_x[0, k],
                    xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k]>= -self.state_constr_limit - eps_x[0, k],
                ]
            else:
                constraints += [
                    xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k]<= self.state_constr_limit,
                    xs_local[self.state_constr_idx] + x_var[self.state_constr_idx, k]>= -self.state_constr_limit,
                ]

        # input constraints  depends on  input_constr_max input_constr_min, here us is already us[u_idx]
        us_local = self.us  # numeric array of length nu
        for k in range(N):
            if self.use_soft_input_constraints:
                constraints += [
                    us_local + u_var[:, k] <= self.input_constr_max + eps_u[0, k],
                    us_local + u_var[:, k] >= self.input_constr_min - eps_u[0, k],
                ]
            else:
                constraints += [
                    us_local + u_var[:, k] <= self.input_constr_max,
                    us_local + u_var[:, k] >= self.input_constr_min,
                ]

        # MPC used
        if self.u_ids[0] == 3:
            mpc_type='Roll'
        elif self.u_ids[0] == 1:
            mpc_type='X'            
        elif self.u_ids[0] == 0:
            mpc_type='Y'            
        elif self.u_ids[0] == 2:
            mpc_type='Z' 

        #no terminal constraints here

        #Cost
        cost = 0
        for k in range(N):
            dx = x_var[:, k] - xref_param
            du = u_var[:, k] - uref_param
            cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)

        # terminal cost
        dxN = x_var[:, N] - xref_param
        cost += cp.quad_form(dxN, P)

        # #Slack costs
        if self.use_soft_state_constraints:
            cost += self.Sx * cp.sum_squares(eps_x)

        if self.use_soft_input_constraints:
            cost += self.Su * cp.sum_squares(eps_u)

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
        self.ocp.solve(solver=cp.PIQP, **self.solver_opts)
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print("MPC problem:", self.ocp.status)

        #retrieve results and convert back to absolute coordinates
        x_opt_dev = np.array(self.x_var.value)
        u_opt_dev = np.array(self.u_var.value)

        #Returns without deviation
        x_traj = x_opt_dev + self.xs.reshape(-1, 1)
        u_traj = u_opt_dev + self.us.reshape(-1, 1)


        # #Saturate input 
        u0_unsat = u_traj[:, 0].flatten()

        u0 = np.clip(u0_unsat, self.input_constr_min, self.input_constr_max)

        if not np.allclose(u0, u0_unsat):
            print(f"[WARN] Input saturated: {u0_unsat} → {u0}")

        u0 = u0.flatten()
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
