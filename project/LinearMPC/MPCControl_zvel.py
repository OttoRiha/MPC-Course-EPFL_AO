import cvxpy as cp
import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base
from scipy.signal import place_poles

class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    #Tunable matrices
    Q = np.diag([20])  # vz 
    R = np.diag([0.1])  # input Pmean

    # state constraints: no limits for Vz
    state_constr_idx = 0
    state_constr_limit = np.inf
		
    # input constraints : min and max allowed power
    input_constr_min = 40.0
    input_constr_max = 80.0

    #Soft constraints
    use_soft_state_constraints = False
    use_soft_input_constraints = False
    Sx = 1e4   # state slack weight
    Su = 1e6   # input slack weight (if enabled)
		
    # only useful for part 5 of the project
    use_mass_deviation_estimator = True
    d_estimate: np.ndarray
    d_gain: float
    L: np.ndarray
    x_hat: np.ndarray
    d_hat: np.ndarray

    A_hat: np.ndarray
    B_hat: np.ndarray
    C_hat: np.ndarray

    u_prev: np.ndarray
    x_est_prev: np.ndarray
    d_param: cp.Parameter

    if use_mass_deviation_estimator:
        def _setup_controller(self) -> None:
            self.u_prev = self.us.flatten().copy()
            self.x_est_prev = np.zeros(self.nx)
            
            # Call parent's setup
            super()._setup_controller()
            
            # Add disturbance parameter
            nx = self.nx
            N = self.N
            self.d_param = cp.Parameter(nx, name='d_disturbance')
            self.d_param.value = np.zeros(nx)
            
            # Rebuild constraints to include disturbance term
            constraints = []
            
            # Initial condition
            constraints += [self.x_var[:, 0] == self.x0_param]
            
            # Dynamics constraints WITH disturbance
            for k in range(N):
                constraints += [
                    self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ (self.u_var[:, k] + self.d_param)
                ]
            
            # No State constraints for vz

            # Input constraints
            us_local = self.us
            for k in range(N):
                constraints += [
                    us_local + self.u_var[:, k] <= self.input_constr_max,
                    us_local + self.u_var[:, k] >= self.input_constr_min
                ]
            
            # NO TERMINAL SET for offset-free tracking (recursive feasibility cannot be guaranteed)
            
            # Rebuild cost and problem
            cost = 0
            K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
            K = -K
            
            for k in range(N):
                dx = self.x_var[:, k] - self.xref_param
                du = self.u_var[:, k] - self.uref_param
                cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)
            
            # Terminal cost
            dxN = self.x_var[:, N] - self.xref_param
            cost += cp.quad_form(dxN, P)
            
            objective = cp.Minimize(cost)
            self.ocp = cp.Problem(objective, constraints)

            # Initialize disturbance estimator
            self.setup_estimator()

        def get_u(
            self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

            # Default targets
            if x_target is None:
                x_target = self.xs.copy()
            if u_target is None:
                u_target = self.us.copy()

            # Update disturbance estimate
            self.update_estimator(x0, self.u_prev)
            
            # Set disturbance parameter to avoid rebuilding problem
            self.d_param.value = self.d_estimate.flatten()
            
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

            # Get solution
            x_opt_dev = np.array(self.x_var.value)
            u_opt_dev = np.array(self.u_var.value)


            # Convert back to absolute coordinates
            x_traj = x_opt_dev + self.xs.reshape(-1, 1)
            u_traj = u_opt_dev + self.us.reshape(-1, 1)
            u0 = u_traj[:, 0].flatten()
            self.u_prev = u0.copy()
            
            # Return full prediction trajectories for visualization
            x_traj = x_opt_dev + self.xs.reshape(-1, 1) if x_opt_dev is not None else np.zeros((self.nx, self.N+1))
            u_traj = u_opt_dev + self.us.reshape(-1, 1) if u_opt_dev is not None else np.zeros((self.nu, self.N))

            return u0, x_traj, u_traj
        
        def setup_estimator(self):
            # FOR PART 5 OF THE PROJECT

            # sizes
            nx = self.nx  
            nu = self.nu  
            nd = 1        # disturbance
            
            # Initialize as 1D arrays
            self.x_hat = self.xs.flatten().copy()
            self.d_hat = np.zeros(nd)
            self.d_estimate = np.zeros(nd)
            
            # Disturbance enters state dynamics additively
            Bd = self.B

            # Measurement model
            C = np.eye(nx)           
            Cd = np.zeros((nx, nd))  
            
            # Augmented system
            self.A_hat = np.block([
                [self.A, Bd],
                [np.zeros((nd, nx)), np.eye(nd)]
            ]) 
            
            self.B_hat = np.vstack([self.B, np.zeros((nd, nu))])  
            
            self.C_hat = np.hstack([C, Cd])  
            
            # Observer poles - mdderate and stable
            poles = np.array([0.6, 0.7])
            res = place_poles(self.A_hat.T, self.C_hat.T, poles)
            self.L = res.gain_matrix.T 
            print(f"L: {(self.L)}")


        def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
            # FOR PART 5 OF THE PROJECT
            nx = self.nx

            # Convert to DEVIATION coordinates
            x_dev = x_data.flatten() - self.xs.flatten()
            u_dev = u_data.flatten() - self.us.flatten()

            # Current augmented estimate
            z_hat = np.concatenate([self.x_hat.flatten(), self.d_hat.flatten()])
            
            # Measurement 
            y_measured = x_dev
            
            # Observer dynamics: z+ = A_hat*z + B_hat*u + L*(y - C_hat*z)
            #                    y = C_hat * z_hat (=x_hat) 
            y_predicted = self.C_hat @ z_hat
            innovation = y_measured - y_predicted
            
            z_hat_next = (self.A_hat @ z_hat + 
                        self.B_hat.flatten() * u_dev.flatten() + 
                        self.L.flatten() * innovation)
            
            # Extract new estimates
            self.x_hat = z_hat_next[:nx]
            self.d_hat = z_hat_next[nx:]
            self.d_estimate = self.d_hat.copy()
        

            print(f"d_estimate: {self.d_estimate}, x_hat: {self.x_hat}, Pavg: {u_data}")
