import numpy as np
import casadi as ca
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    def __init__(self, rocket,H, xs, us, Ts):
        # store parameters
        self.rocket = rocket
        self.H=H
        self.xs=xs
        self.us=us
        self.N = int(H / Ts)
        self.Ts = Ts
        

        # symbolic dynamics (continuous time)
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        # dimensions
        self.nx = 12
        self.nu = 4

        # # steady state (trim)
        # if not hasattr(rocket, "xs") or not hasattr(rocket, "us"):
        #     rocket.trim()
        # self.xs = rocket.xs
        # self.us = rocket.us

        self._setup_controller()

    def _rk4(self, x, u):
        h = self.Ts
        k1 = self.f(x, u)
        k2 = self.f(x + h/2 * k1, u)
        k3 = self.f(x + h/2 * k2, u)
        k4 = self.f(x + h * k3, u)
        return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def _euler(self, x, u):
        h=self.Ts
        return x + h * self.f(x, u)

    def _setup_controller(self) -> None:

        opti = ca.Opti()

        use_soft_constr=False
        use_euler=True #uses euler if true, uses rk4 if false

        # decision variables
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        # parameters
        X0 = opti.parameter(self.nx)

        # cost weights (tune if needed)
        # Q = ca.diag(ca.DM([
        #     0.1, 0.1, 0.1,        # angular rates
        #     10, 10, 10,           # angles
        #     0.1, 0.1, 0.5,       # velocities
        #     1, 1, 1             # positions
        # ]))
        Q = 0.8*ca.diag(ca.DM([
            0.5, 0.5, 0.2,        # angular rates
            80, 80, 80,           # angles (gamma/roll much higher)
            3, 3, 5,              # velocities (vx, vy, vz)
            50, 50, 80            # positions (x, y, z)
        ]))

        R = ca.diag(ca.DM([10, 10, 0.1, 0.1])) 
        if use_soft_constr:
            # slack variables for state constraints
            S_beta = opti.variable(1, self.N + 1)
            S_z    = opti.variable(1, self.N + 1)
            W_slack = 1e3 #penalty in optimisation
            opti.subject_to(S_beta >= 0)
            opti.subject_to(S_z    >= 0)

        cost = 0

        # initial condition
        opti.subject_to(X[:, 0] == X0)

        # dynamics + running cost
        for k in range(self.N):
            if use_euler:
                opti.subject_to(X[:, k+1] == self._euler(X[:, k], U[:, k]))
            else:    
                opti.subject_to(X[:, k+1] == self._rk4(X[:, k], U[:, k]))

            dx = X[:, k] - self.xs
            du = U[:, k] - self.us
            cost += dx.T @ Q @ dx + du.T @ R @ du
            # #Slack cost
            # if use_soft_constr:
            #     cost += W_slack * ca.sumsqr(S_beta)
            #     cost += W_slack * ca.sumsqr(S_z)
        if use_soft_constr:
            cost += W_slack * ca.sumsqr(S_beta) + W_slack * ca.sumsqr(S_z)

        
        # terminal cost
        dxN = X[:, self.N] - self.xs
        Qf = 100 * Q
        cost += dxN.T @ Qf @ dxN

        # -----------------------
        # INPUT CONSTRAINTS
        # -----------------------
        d1_max = np.deg2rad(15)
        d2_max = np.deg2rad(15)

        opti.subject_to(opti.bounded(-d1_max, U[0, :], d1_max))   # δ1
        opti.subject_to(opti.bounded(-d2_max, U[1, :], d2_max))   # δ2
        opti.subject_to(opti.bounded(10.0, U[2, :], 90.0))        # Pavg
        opti.subject_to(opti.bounded(-20.0, U[3, :], 20.0))       # Pdiff

        # -----------------------
        # STATE CONSTRAINTS
        # -----------------------
        beta_idx = 4
        z_idx = 11
        beta_max = np.deg2rad(80)
        if use_soft_constr:
            opti.subject_to(X[beta_idx, :] <=  beta_max + S_beta)
            opti.subject_to(X[beta_idx, :] >= -beta_max - S_beta)
            opti.subject_to(X[z_idx, :] >= -S_z)
        else:
            opti.subject_to(opti.bounded(-beta_max, X[beta_idx, :], beta_max))
            opti.subject_to(X[z_idx, :] >= 0)
        
        #Problem setup
        opti.minimize(cost)
        opts = {
            "expand": True,
            "ipopt": {
                "print_level": 0,
                "max_iter": 800,                 # don’t allow 3000 in closed loop
                "tol": 1e-3,
                "acceptable_tol": 1e-2,
                "acceptable_iter": 5,            # allow early exit if “good enough”
                "hessian_approximation": "limited-memory",

                # warm start settings
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6,
                "mu_init": 1e-2
            }

        }
        

        opti.solver("ipopt", opts)

        # store objects
        self.opti = opti
        self.X = X
        self.U = U
        self.X0 = X0


    def get_u(
        self, t0: float, x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # set parameter
        self.opti.set_value(self.X0, x0)

        # warm start from previous solution
        if hasattr(self, "x_guess"):
            self.opti.set_initial(self.X, self.x_guess)
            self.opti.set_initial(self.U, self.u_guess)
        else:
            Xinit = np.tile(x0.reshape(-1,1), (1, self.N+1))
            Uinit = np.tile(self.us.reshape(-1,1), (1, self.N))
            self.opti.set_initial(self.X, Xinit)
            self.opti.set_initial(self.U, Uinit)


        # solve NMPC
        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            print("NMPC solver failed at t =", t0)
            print("x0 =", x0)
            print("X (debug):", self.opti.debug.value(self.X))
            print("U (debug):", self.opti.debug.value(self.U))

            # fallback u0
            if hasattr(self, "u_guess"):
                u0 = self.u_guess[:, 0]
            else:
                u0 = np.array(self.us).reshape(-1)

            # fallback trajectories
            x_ol = np.tile(x0.reshape(-1,1), (1, self.N+1))
            u_ol = np.tile(u0.reshape(-1,1), (1, self.N))
            t_ol = t0 + self.Ts * np.arange(self.N + 1)

            return u0, x_ol, u_ol, t_ol



        # extract trajectories
        x_ol = sol.value(self.X)
        u_ol = sol.value(self.U)
        # after solving
        self.x_guess = np.hstack([x_ol[:, 1:], x_ol[:, -1:]])   # shift state guess
        self.u_guess = np.hstack([u_ol[:, 1:], u_ol[:, -1:]])   # repeat last control


        # first input
        u0 = u_ol[:, 0]

        # time vector
        t_ol = t0 + self.Ts * np.arange(self.N + 1)

        return u0, x_ol, u_ol, t_ol