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
    R = np.diag([0.1])          # [Pavg] - smaller to allow more control effort

    # State constraints: z >= 0
    state_constr_idx = 1  # z position
    state_constr_limit = np.inf  # No upper limit
    
    # Input constraints: 40 <= Pavg <= 80
    input_constr_min = 40.0
    input_constr_max = 80.0

    # No soft constraints for robust tube MPC
    use_soft_state_constraints = False
    use_soft_input_constraints = False
    Sx = 0.0
    Su = 0.0

    # Disturbance bounds - REALISTIC VALUES
    # Disturbance only affects velocity (vz), not position directly
    # Model mismatch should be small, on order of 0.1-0.5 m/s²
    w_vz_min = -0.5  # [m/s] per time step (Ts = 0.05s means -10 m/s² acceleration disturbance)
    w_vz_max = 0.5   # [m/s] per time step
    w_z_min = 0.0    # No direct disturbance on position
    w_z_max = 0.0    # No direct disturbance on position

    def _setup_controller(self) -> None:
        """Setup tube MPC controller with robust invariant sets"""
        nx = self.nx
        nu = self.nu
        N = self.N

        print(f"\n=== Setting up Tube MPC for Z-dimension ===")
        print(f"Sampling time Ts = {self.Ts:.4f} s")
        print(f"Horizon N = {N} steps ({self.H:.2f} s)")
        
        # Compute LQR feedback gain K
        K, Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        self.K = K
        self.Qf = Qf
        A_cl = self.A + self.B @ K
        
        # Check stability
        eigs = np.linalg.eigvals(A_cl)
        print(f"\nClosed-loop eigenvalues: {eigs}")
        print(f"Max eigenvalue magnitude: {np.max(np.abs(eigs)):.4f}")

        print(f"LQR gain K = {K}")

        # Input disturbance bounds
        w_input_min = -15.0
        w_input_max = 5.0
        
        print(f"\nInput disturbance: w_input ∈ [{w_input_min}, {w_input_max}]")
        
        # State disturbance: W_state = B * W_input
        w_state_min = self.B.flatten() * w_input_min
        w_state_max = self.B.flatten() * w_input_max
        
        W = Polyhedron.from_Hrep(
            A=np.vstack((np.eye(nx), -np.eye(nx))),
            b=np.array([
                max(w_state_max[0], w_state_min[0]),
                max(w_state_max[1], w_state_min[1]),
                -min(w_state_min[0], w_state_max[0]),
                -min(w_state_min[1], w_state_max[1])
            ])
        )
        self.W = W
        
        print(f"State disturbance (B*w): vz ∈ [{min(w_state_min[0], w_state_max[0]):.4f}, {max(w_state_max[0], w_state_min[0]):.4f}]")
        print(f"                         z  ∈ [{min(w_state_min[1], w_state_max[1]):.4f}, {max(w_state_max[1], w_state_min[1]):.4f}]")

        # Compute minimal robust invariant set E
        self.E = self._min_robust_invariant_set(A_cl, W, max_iter=100)
        
        # Print E bounds for diagnostics
        E_bbox = Polyhedron.bounding_box(self.E)
        print(f"\nMinimal invariant set E bounding box:")
        print(f"  vz ∈ [{-E_bbox.b[2]:.4f}, {E_bbox.b[0]:.4f}]")
        print(f"  z  ∈ [{-E_bbox.b[3]:.4f}, {E_bbox.b[1]:.4f}]")
        
        # Define constraints
        z_steady_state = self.xs[1]
        safety_margin = 0.5
        
        # State constraint: z >= safety_margin (absolute)
        # In deviation: z_dev >= safety_margin - z_steady_state
        X = Polyhedron.from_Hrep(
            A=np.array([[0, -1]]),
            b=np.array([z_steady_state - safety_margin])
        )
        self.X = X
        
        print(f"\nState constraint (absolute): z >= {safety_margin:.2f} m")
        print(f"State constraint (deviation): z_dev >= {-(z_steady_state - safety_margin):.2f} m")

        # Input constraints
        U = Polyhedron.from_Hrep(
            A=np.array([[1], [-1]]),
            b=np.array([self.input_constr_max - self.us[0], -(self.input_constr_min - self.us[0])])
        )
        self.U = U
        
        print(f"Input constraint: {self.input_constr_min:.2f} <= u <= {self.input_constr_max:.2f}")

        # Compute tightened constraints MANUALLY to avoid Pontryagin difference issues
        # X_tilde = {x | x + e ∈ X for all e ∈ E}
        # For X = {x | A_x * x <= b_x} and E with support function:
        # X_tilde = {x | A_x * x <= b_x - support_E(A_x)}
        
        print(f"\nComputing tightened constraints...")
        
        # Tightened state constraints
        X_tilde_A = X.A.copy()
        X_tilde_b = X.b.copy()
        for i in range(X.A.shape[0]):
            support_val = self.E.support(X.A[i, :])
            X_tilde_b[i] -= support_val
            print(f"  State constraint {i}: support_E = {support_val:.4f}, tightened by {support_val:.4f}")
        
        X_tilde = Polyhedron.from_Hrep(X_tilde_A, X_tilde_b)
        self.X_tilde = X_tilde
        
        if X_tilde.is_empty:
            print("  WARNING: X_tilde is empty! E is too large for state constraints.")
            print("  Relaxing safety margin...")
            # Relax constraint
            X_tilde_b = X_tilde_b + np.max(np.abs(X_tilde_b)) * 0.5
            X_tilde = Polyhedron.from_Hrep(X_tilde_A, X_tilde_b)
            self.X_tilde = X_tilde
        
        # Tightened input constraints: U_tilde = U - K*E
        KE = self.E.affine_map(K)
        U_tilde_A = U.A.copy()
        U_tilde_b = U.b.copy()
        for i in range(U.A.shape[0]):
            support_val = KE.support(U.A[i, :])
            U_tilde_b[i] -= support_val
            print(f"  Input constraint {i}: support_KE = {support_val:.4f}, tightened by {support_val:.4f}")
        
        U_tilde = Polyhedron.from_Hrep(U_tilde_A, U_tilde_b)
        self.U_tilde = U_tilde
        
        if U_tilde.is_empty:
            print("  WARNING: U_tilde is empty! E is too large for input constraints.")
            # This is a serious problem - tube MPC won't work
            U_tilde_b = U_tilde_b + np.max(np.abs(U_tilde_b)) * 0.5
            U_tilde = Polyhedron.from_Hrep(U_tilde_A, U_tilde_b)
            self.U_tilde = U_tilde

        # Terminal set
        print(f"\nComputing terminal set...")
        KU_tilde = Polyhedron.from_Hrep(U_tilde.A @ K, U_tilde.b)
        
        if not X_tilde.is_empty and not KU_tilde.is_empty:
            X_tilde_and_KU_tilde = X_tilde.intersect(KU_tilde)
        elif not KU_tilde.is_empty:
            X_tilde_and_KU_tilde = KU_tilde
        else:
            print("  WARNING: Both X_tilde and KU_tilde problematic!")
            X_tilde_and_KU_tilde = Polyhedron.from_Hrep(
                A=np.vstack((np.eye(nx), -np.eye(nx))),
                b=0.1 * np.ones(2*nx)
            )
        
        if X_tilde_and_KU_tilde.is_empty:
            print("  WARNING: Constraint intersection is empty!")
            X_tilde_and_KU_tilde = Polyhedron.from_Hrep(
                A=np.vstack((np.eye(nx), -np.eye(nx))),
                b=0.1 * np.ones(2*nx)
            )
        
        self.Xf_tilde = self._max_robust_invariant_set(A_cl, X_tilde_and_KU_tilde, W, max_iter=50)
        
        if self.Xf_tilde.is_empty:
            print("  WARNING: Terminal set is empty! Using fallback.")
            self.Xf_tilde = Polyhedron.from_Hrep(
                A=np.vstack((np.eye(nx), -np.eye(nx))),
                b=0.1 * np.ones(2*nx)
            )

        # CVXPY formulation
        z_var = cp.Variable((nx, N + 1), name='z')
        v_var = cp.Variable((nu, N), name='v')
        x0_param = cp.Parameter((nx,), name='x0')

        self.z_var = z_var
        self.v_var = v_var
        self.x0_param = x0_param

        # Cost
        cost = 0
        for k in range(N):
            cost += cp.quad_form(z_var[:, k], self.Q)
            cost += cp.quad_form(v_var[:, k], self.R)
        cost += cp.quad_form(z_var[:, N], Qf)

        # Constraints
        constraints = []

        # Initial condition: x0 in z0 + E
        constraints.append(self.E.A @ (x0_param - z_var[:, 0]) <= self.E.b)

        # Dynamics
        for k in range(N):
            constraints.append(z_var[:, k + 1] == self.A @ z_var[:, k] + self.B @ v_var[:, k])

        # Tightened constraints
        if not X_tilde.is_empty:
            for k in range(N):
                constraints.append(X_tilde.A @ z_var[:, k] <= X_tilde.b)

        if not U_tilde.is_empty:
            for k in range(N):
                constraints.append(U_tilde.A @ v_var[:, k] <= U_tilde.b)

        # Terminal constraint
        constraints.append(self.Xf_tilde.A @ z_var[:, N] <= self.Xf_tilde.b)

        objective = cp.Minimize(cost)
        self.ocp = cp.Problem(objective, constraints)
        
        print(f"\n=== Tube MPC Setup Complete ===")
        print(f"  - E: {self.E.A.shape[0]} constraints")
        print(f"  - X_tilde: {self.X_tilde.A.shape[0]} constraints, empty={self.X_tilde.is_empty}")
        print(f"  - U_tilde: {self.U_tilde.A.shape[0]} constraints, empty={self.U_tilde.is_empty}")
        print(f"  - Xf_tilde: {self.Xf_tilde.A.shape[0]} constraints, empty={self.Xf_tilde.is_empty}")
        
        self._plot_sets()

    def _min_robust_invariant_set(self, A_cl: np.ndarray, W: Polyhedron, max_iter: int = 50) -> Polyhedron:
        """Compute minimal robust invariant set"""
        nx = A_cl.shape[0]
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep()
            
            if np.linalg.norm(A_cl_ith_power, ord=2) < 1e-2:
                print(f'Minimal robust invariant set E converged after {itr} iterations.')
                break
            
            Omega = Omega_next
            itr += 1
        
        if itr == max_iter:
            print(f'Minimal robust invariant set E did NOT converge after {max_iter} iterations.')
        
        return Omega_next

    def _robust_pre_set(self, A_cl: np.ndarray, Omega: Polyhedron, W: Polyhedron) -> Polyhedron:
        """Compute robust pre-set"""
        b_pre = Omega.b.copy()
        for i in range(Omega.b.shape[0]):
            b_pre[i] -= W.support(Omega.A[i, :])
        pre_Omega = Polyhedron.from_Hrep(Omega.A @ A_cl, b_pre)
        return pre_Omega

    def _max_robust_invariant_set(self, A_cl: np.ndarray, Omega: Polyhedron, W: Polyhedron, max_iter: int = 50) -> Polyhedron:
        """Compute maximal robust invariant set"""
        iter_count = 0
        Omega_inf = Polyhedron.from_Hrep(Omega.A, Omega.b)

        while iter_count < max_iter:
            iter_count += 1
            Omega_inf_pre = self._robust_pre_set(A_cl, Omega_inf, W)
            Omega_inf_next = Omega_inf.intersect(Omega_inf_pre)
            Omega_inf_next.minHrep()

            if Omega_inf_next == Omega_inf:
                print(f'Maximal robust invariant set Xf_tilde converged after {iter_count} iterations.')
                break

            Omega_inf = Omega_inf_next

        if iter_count == max_iter:
            print(f'Maximal robust invariant set Xf_tilde did NOT converge after {max_iter} iterations.')

        return Omega_inf

    def _plot_sets(self):
        """Plot the invariant sets - robust version"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot E (minimal invariant set)
        ax1.set_title('Minimal Robust Invariant Set E')
        ax1.set_xlabel('vz [m/s]')
        ax1.set_ylabel('z [m]')
        
        # E should be bounded, but check dimension
        if self.E.dim == 2:
            try:
                self.E.plot(ax1, color='red', opacity=0.5, label=r'$\mathcal{E}$')
            except:
                # Fallback: project to 2D
                E_proj = self.E.projection((0, 1))
                if not E_proj.is_empty:
                    E_proj.plot(ax1, color='red', opacity=0.5, label=r'$\mathcal{E}$')
        ax1.legend()
        ax1.grid(True)
        
        # Plot terminal set Xf_tilde
        ax2.set_title('Terminal Set Xf_tilde')
        ax2.set_xlabel('vz [m/s]')
        ax2.set_ylabel('z [m]')
        
        # Try to plot Xf_tilde
        try:
            if self.Xf_tilde.is_empty:
                print("Warning: Terminal set Xf_tilde is empty!")
                ax2.text(0.5, 0.5, 'Empty Set', ha='center', va='center', transform=ax2.transAxes)
            elif self.Xf_tilde.dim >= 2:
                # Project to first 2 dimensions if needed
                Xf_proj = self.Xf_tilde.projection((0, 1)) if self.Xf_tilde.dim > 2 else self.Xf_tilde
                if not Xf_proj.is_empty:
                    Xf_proj.plot(ax2, color='blue', opacity=0.5, label=r'$\tilde{\mathcal{X}}_f$')
            elif self.Xf_tilde.dim == 1:
                # 1D set - plot as interval
                B = Polyhedron.bounding_box(self.Xf_tilde)
                H = B.A
                h = B.b
                lb = -np.inf
                ub = np.inf
                for i in range(H.shape[0]):
                    if H[i, 0] == 1:
                        ub = min(ub, h[i])
                    elif H[i, 0] == -1:
                        lb = max(lb, -h[i])
                ax2.plot([lb, ub], [0, 0], linewidth=4, color='blue', label=r'$\tilde{\mathcal{X}}_f$')
        except Exception as e:
            print(f"Note: Could not plot terminal set: {e}")
            ax2.text(0.5, 0.5, f'Plotting error:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=8)
        
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-2, 5)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print constraint information
        print("\n=== Constraint Information ===")
        print(f"Minimal invariant set E:")
        print(f"  Dimension: {self.E.dim}")
        print(f"  H-rep shape: A={self.E.A.shape}, b={self.E.b.shape}")
        
        print(f"\nTerminal set Xf_tilde:")
        print(f"  Dimension: {self.Xf_tilde.dim}")
        print(f"  H-rep shape: A={self.Xf_tilde.A.shape}, b={self.Xf_tilde.b.shape}")
        print(f"  Is empty: {self.Xf_tilde.is_empty}")
        print(f"  Is bounded: {self.Xf_tilde.is_bounded}")
        
        print(f"\nTightened input constraint U_tilde:")
        print(f"  H-rep: A={self.U_tilde.A}, b={self.U_tilde.b}")
        try:
            # Try to get vertices
            V = self.U_tilde.V
            if V is not None and len(V) > 0:
                print(f"  Vertices (deviation): {V.flatten()}")
                print(f"  Vertices (absolute): {V.flatten() + self.us[0]}")
        except:
            # Compute bounds from constraints
            if self.U_tilde.A.shape[0] >= 2:
                bounds = []
                for i in range(self.U_tilde.A.shape[0]):
                    if abs(self.U_tilde.A[i, 0]) > 1e-10:
                        bound = self.U_tilde.b[i] / self.U_tilde.A[i, 0]
                        bounds.append(bound)
                if bounds:
                    print(f"  Bounds (deviation): [{min(bounds):.4f}, {max(bounds):.4f}]")
                    print(f"  Bounds (absolute): [{min(bounds) + self.us[0]:.4f}, {max(bounds) + self.us[0]:.4f}]")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve tube MPC and return control input"""
        
        # Work in deviation coordinates
        x0_dev = x0 - self.xs
        
        # Set parameter
        self.x0_param.value = x0_dev
        
        # Solve
        self.ocp.solve(solver=cp.PIQP, verbose=False)
        
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Tube MPC Z: Solver status: {self.ocp.status}")
            # Return safe fallback
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj
        
        # Get nominal solution
        z_opt = np.array(self.z_var.value)
        v_opt = np.array(self.v_var.value)
        
        # Apply tube MPC control law: u = v + K(x - z)
        z0 = z_opt[:, 0]
        v0 = v_opt[:, 0]
        u0_dev = v0 + self.K @ (x0_dev - z0)
        u0 = u0_dev + self.us
        
        # Convert trajectories back to absolute coordinates
        x_traj = z_opt + self.xs.reshape(-1, 1)
        u_traj = v_opt + self.us.reshape(-1, 1)
        
        return u0.flatten(), x_traj, u_traj