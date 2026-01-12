import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # [Pavg]

    # Tunable matrices for tube MPC
    Q = np.diag([20.0, 50.0])  # [vz, z] - heavier weight on position
    R = np.diag([1])          # [Pavg] - smaller to allow more control effort

    # State constraints: z >= 0
    state_constr_idx = 1  # z position
    state_constr_min = 0  # 0 is the lower limit
    state_constr_max = 20 # CVXPY complains about inf
    
    # Input constraints: 40 <= Pavg <= 80
    input_constr_min = 40.0
    input_constr_max = 80.0

    # Uncertainity Constraints
    w_min = -15.0
    w_max = 5



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
        self.K = K
        print(f"K: {K}")
        # MPC used
        if self.u_ids[0] == 3:
            mpc_type='Roll'
        elif self.u_ids[0] == 1:
            mpc_type='X'            
        elif self.u_ids[0] == 0:
            mpc_type='Y'            
        elif self.u_ids[0] == 2:
            mpc_type='Z' 


        #Variable definition
        x_var = cp.Variable((nx, N + 1), name='x')
        u_var = cp.Variable((nu, N), name='u')
        x0_param = cp.Parameter((nx,), name='x0')
        xref_param = cp.Parameter(nx, value=np.zeros(nx))
        uref_param = cp.Parameter(nu, value=np.zeros(nu))

        
        #Invariant set computation
        A_cl = self.A + self.B @ K
        U = Polyhedron.from_Hrep(
            np.array([[-1], [1]]),
            np.array([-self.input_constr_min, self.input_constr_max])
        )
        
        H = np.zeros((2, nx))
        H[0, self.state_constr_idx] = -1
        H[1, self.state_constr_idx] = 1
        h = np.array([self.state_constr_min, self.state_constr_max])
        X = Polyhedron.from_Hrep(H, h)

        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        # Compute the terminal set for nominal mpc
        X_and_KU = X.intersect(KU)
        Xf = MPCControl_base.max_invariant_set(A_cl, X_and_KU, mpc_type=mpc_type)

        #min robust invariant set computation

        W = Polyhedron.from_Hrep(np.array([[-1], [1]]),
                                 np.array([-self.w_min, self.w_max]))
        # Map to state space: Bd @ w
        W_state = W.affine_map(self.B)  # Now 2D polyhedron

        E = MPCControl_base.min_robust_invariant_set(A_cl, W_state)

         # visualization
        # fig3, ax3 = plt.subplots(1, 1)
        # X.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{X}$')
        # W.plot(ax3, color='b', opacity=0.5, label=r'$\mathcal{W}$') #'$\tilde{\mathcal{X}}_f$')
        # E.plot(ax3, color='b', opacity=0.5, label=r'$\mathcal{E}$') 
        # plt.legend()
        # plt.show()

        # tightened state constraints
        X_tilde = X - E

        # tightened input constraints
        KE = E.affine_map(K) 

        U_tilde = U - KE
        
        if X.is_empty:
            print("CRITICAL: X is empty!")
        if U.is_empty:
            print("CRITICAL: U is empty!")
        if W.is_empty:
            print("CRITICAL: W is empty!")
        if E.is_empty:
            print("CRITICAL: E is empty!")
        if X_tilde.is_empty:
            print("CRITICAL: X_tilde is empty! Make Q larger or W smaller.")
            print(f"X vertices: {X.V if hasattr(X, 'V') and X.V is not None else 'N/A'}")
            print(f"E vertices: {E.V if hasattr(E, 'V') and E.V is not None else 'N/A'}")
        if U_tilde.is_empty:
            print("CRITICAL: U_tilde is empty! Make R larger or W smaller.")

        print(f"X_tilde vertices:\n{X_tilde.V}")
        print(f"U_tilde vertices:\n{U_tilde.V}")
        print(f"U_tilde has {U_tilde.A.shape[0]} constraints")

        # move into deviate coordinates H * (xs + dx) <= k  =>  H * dx <= k - H * xs
        h_x_delta = X_tilde.b - X_tilde.A @ self.xs
        X_tilde_delta = Polyhedron.from_Hrep(X_tilde.A, h_x_delta)
        # H * (us + du) <= k  =>  H * du <= k - H * us
        h_u_delta = U_tilde.b - U_tilde.A @ self.us
        U_tilde_delta = Polyhedron.from_Hrep(U_tilde.A, h_u_delta)

        if not X_tilde_delta.contains(np.zeros(nx)):
            print("WARNING: X_tilde_dev doesn't contain origin! Steady state might be infeasible.")
            
        if not U_tilde_delta.contains(np.zeros(nu)):
            print("WARNING: U_tilde_dev doesn't contain origin! Steady state might be infeasible.")
        
        # Compute the terminal set for tube mpc
        X_tilde_and_KU_tilde_delta = X_tilde_delta.intersect(Polyhedron.from_Hrep(U_tilde_delta.A @ K, U_tilde_delta.b))
        if X_tilde_and_KU_tilde_delta.is_empty:
            print("CRITICAL: X_tilde ∩ K*U_tilde is empty! Cannot compute terminal set.")


        Xf_tilde = MPCControl_base.max_invariant_set(A_cl, X_tilde_and_KU_tilde_delta, mpc_type=mpc_type)
        if Xf_tilde.is_empty:
            print("CRITICAL: Xf_tilde is empty! Terminal set computation failed.")
        print(f"Terminal set Xf_tilde computed with {Xf_tilde.A.shape[0]} constraints")
        
        
        

        #TODO: Visualisation of these constraints

        #                 visualization
        # fig3, ax3 = plt.subplots(1, 1)
        # X_tilde.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{X}_f$')
        # U_tilde.plot(ax3, color='b', opacity=0.5, label=r'$\tilde{\mathcal{X}}_f$')
        # plt.legend()
        # plt.show()

        #Constraint definition
        constraints = []

        # initial condition: x0 in z0 + E 
        # constraints += [E.A @ (x0_param - x_var[:, 0]) <= E.b]
        constraints += [x_var[:, 0] == x0_param]

        # dynamics constraints
        for k in range(N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]

        # state constraints: use tightened constraints in deviation coordinates
        for k in range(N + 1):
            constraints += [X_tilde_delta.A @ x_var[:, k] <= X_tilde_delta.b]

        # input constraints: use tightened constraints in deviation coordinates
        for k in range(N):
            constraints += [U_tilde_delta.A @ u_var[:, k] <= U_tilde_delta.b]

        if Xf_tilde.is_empty:
            print("CRITICAL: Xf_tilde is empty!")
        else:
            # Terminal Constraints: use tightened terminal set in deviation coordinates
            constraints.append(Xf_tilde.A @ x_var[:, -1] <= Xf_tilde.b)


                # visualization
        fig3, ax3 = plt.subplots(1, 1)
        E.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{E}$')
        Xf_tilde.plot(ax3, color='b', opacity=0.5, label=r'$\tilde{\mathcal{X}}_f$')
        plt.legend()
        plt.show()

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





        # self.ocp = ...

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

         # Default targets
        if x_target is None:
            x_target = self.xs.copy()
        if u_target is None:
            u_target = self.us.copy()

        # Work in deviation coordinates for the solver:
        x0_dev = x0 - self.xs
        xref_dev = x_target - self.xs
        uref_dev = u_target - self.us

        # Set CVXPY parameters
        self.x0_param.value = x0_dev
        self.xref_param.value = xref_dev
        self.uref_param.value = uref_dev

        # Solve
        self.ocp.solve(solver=cp.PIQP, **self.solver_opts)
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"MPC problem status: {self.ocp.status}")

        # Get nominal trajectories (in deviation coords)
        z_opt_dev = np.array(self.x_var.value)  # nominal state trajectory
        v_opt_dev = np.array(self.u_var.value)  # nominal control trajectory
        
        # Tube MPC: Apply ancillary feedback controller
        # u0 = v0 + K(x0 - z0)
        K = self.K  # Store K as attribute in _setup_controller
        u0_dev = v_opt_dev[:, 0] + K @ (x0_dev - z_opt_dev[:, 0])
        
        # Convert to absolute coordinates
        u0 = u0_dev + self.us
        u0 = u0.flatten()
        

        # Saturate
        u0 = np.clip(u0, self.input_constr_min, self.input_constr_max)
        
        # Return full prediction trajectories for visualization
        x_traj = z_opt_dev + self.xs.reshape(-1, 1) if z_opt_dev is not None else np.zeros((self.nx, self.N+1)) # nominal trajectory
        u_traj = v_opt_dev + self.us.reshape(-1, 1) if v_opt_dev is not None else np.zeros((self.nu, self.N))  # nominal control trajectory

        # u0 = ...
        # x_traj = ...
        # u_traj = ...

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    