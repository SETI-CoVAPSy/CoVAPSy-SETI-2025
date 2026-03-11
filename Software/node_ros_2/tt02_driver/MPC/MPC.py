import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# Colors
PREDICTION = '#BA4A00'

##################
# MPC Controller #
##################


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, SolverSettings=None):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        """

        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal

        # Model
        self.model = model

        # Dimensions
        self.nx = self.model.n_states
        self.nu = 2

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Maximum lateral acceleration
        self.ay_max = ay_max

        default_solver_settings = {
            'verbose': False,
            'warm_start': True,
            'polish': False,
            'adaptive_rho': True,
            'max_iter': 10000,
            'eps_abs': 1e-3,
            'eps_rel': 1e-3,
        }
        if SolverSettings is not None:
            default_solver_settings.update(SolverSettings)
        self.solver_settings = default_solver_settings

        # Current control and prediction
        self.current_prediction = None

        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0

        # Log throttling counters to avoid flooding terminal on repeated
        # infeasible/suboptimal solver outcomes.
        self._fallback_count = 0
        self._suboptimal_count = 0
        self._status_log_every = 25

        # Current control signals
        self.current_control = np.zeros((self.nu*self.N))
        # Last applied control [v, delta] used as robust fallback.
        self.last_applied_control = np.zeros(self.nu)

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()
        self._problem_initialized = False
        self.P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
             sparse.kron(sparse.eye(self.N), self.R)], format='csc')

    def _get_fallback_control(self):
        """Return a safe fallback control when optimization is invalid."""
        if np.all(np.isfinite(self.last_applied_control)):
            return np.array(self.last_applied_control, dtype=float)
        return np.array([0.0, 0.0])

    def _log_fallback(self, status_raw: str):
        """Log fallback events sparsely to keep output readable."""
        self._fallback_count += 1
        self._suboptimal_count = 0
        if self._fallback_count == 1 or self._fallback_count % self._status_log_every == 0:
            print(
                f"OSQP failed with status='{status_raw}'. Using fallback control "
                f"(count={self._fallback_count})."
            )

    def _log_suboptimal(self, status_raw: str):
        """Log suboptimal events sparsely to avoid noisy repeated warnings."""
        self._suboptimal_count += 1
        if self._suboptimal_count == 1 or self._suboptimal_count % self._status_log_every == 0:
            print(
                f"OSQP status='{status_raw}', using best available iterate "
                f"(count={self._suboptimal_count})."
            )

    def _log_recovery_if_needed(self):
        """Emit a compact recovery message after prolonged degraded operation."""
        if self._fallback_count >= self._status_log_every:
            print(f"OSQP recovered after {self._fallback_count} fallback step(s).")
        self._fallback_count = 0
        self._suboptimal_count = 0

    def _init_problem(self):
        """
        Initialize optimization problem for current time step.
        """

        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # LTV System Matrices
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
        # Reference vector for state and input variables
        ur = np.zeros(self.nu*self.N)
        xr = np.zeros(self.nx*(self.N+1))
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(self.N * self.nx)
        # Dynamic state constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Dynamic input constraints
        umax_dyn = np.kron(np.ones(self.N), umax)
        # Get curvature predictions from previous control signals
        deltas = np.array(self.current_control[1::2])
        if len(deltas) > 1:
            deltas_pred = np.append(deltas[1:], deltas[-1])
        else:
            deltas_pred = deltas
        kappa_pred = np.tan(deltas_pred) / self.model.length

        # Iterate over horizon
        for n in range(self.N):

            # Get information about current waypoint
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            v_ref = current_waypoint.v_ref

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[(n+1) * self.nx: (n+2)*self.nx, n * self.nx:(n+1)*self.nx] = A_lin
            B[(n+1) * self.nx: (n+2)*self.nx, n * self.nu:(n+1)*self.nu] = B_lin

            # Set reference for input signal
            ur[n*self.nu:(n+1)*self.nu] = np.array([v_ref, kappa_ref])
            # Compute equality constraint offset (B*ur)
            uq[n * self.nx:(n+1)*self.nx] = B_lin.dot(np.array
                                            ([v_ref, kappa_ref])) - f

            # Constrain maximum speed based on predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            if vmax_dyn < umax_dyn[self.nu*n]:
                # fonction pour limiter vmax_dyn avec la vitesse minimale (index 0 de umin)
                umax_dyn[self.nu*n] = max(vmax_dyn, umin[0])

        # Compute dynamic constraints on e_y
        lateral_margin = self.model.safety_margin
        ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id+1, self.N, lateral_margin, lateral_margin)

        # Ensure initial state feasibility (k=0) without hacking the whole horizon
        # If current state is out of bounds, relax ONLY the first step constraint.
        if self.model.spatial_state.e_y > ub[0]: ub[0] = self.model.spatial_state.e_y + 1e-4
        if self.model.spatial_state.e_y < lb[0]: lb[0] = self.model.spatial_state.e_y - 1e-4

        xmin_dyn[self.nx::self.nx] = lb
        xmax_dyn[self.nx::self.nx] = ub

        # Set reference for state as center-line of drivable area
        xr[self.nx::self.nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(self.N + 1),
                         -sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([xmin_dyn,
                           np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        # Get upper and lower bound vectors for inequality constraints
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        q = np.hstack(
            [-np.tile(np.diag(self.Q.A), self.N) * xr[:-self.nx],
             -self.QN.dot(xr[-self.nx:]),
             -np.tile(np.diag(self.R.A), self.N) * ur])

        # Initialize or update optimizer
        if not self._problem_initialized:
            self.optimizer.setup(P=self.P, q=q, A=A, l=l, u=u, **self.solver_settings)
            self._problem_initialized = True
        else:
            self.optimizer.update(l=l, u=u, q=q, Ax=A.data)

    def get_control(self):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # Number of state variables
        nx = self.model.n_states
        nu = 2

        # Update current waypoint unless an external module (ROS node) manages
        # waypoint synchronization explicitly.
        if not getattr(self.model, "external_waypoint_sync", False):
            self.model.get_current_waypoint()

        # Update spatial state
        self.model.spatial_state = self.model.t2s(reference_state=
            self.model.temporal_state, reference_waypoint=
            self.model.current_waypoint)

        # Initialize optimization problem
        self._init_problem()

        # Solve optimization problem
        dec = self.optimizer.solve()

        status_raw = str(getattr(dec.info, "status", "unknown"))
        status = status_raw.lower()
        allow_suboptimal = "maximum iterations reached" in status
        if dec.x is None or (("solved" not in status) and not allow_suboptimal):
            self._log_fallback(status_raw)
            u = self._get_fallback_control()
            self.infeasibility_counter += 1
            if self.infeasibility_counter >= self.N:
                self.infeasibility_counter = self.N - 1
            return u
        if allow_suboptimal:
            self._log_suboptimal(status_raw)

        try:
            # Get control signals
            control_signals = np.asarray(dec.x[-self.N*nu:], dtype=float)
            if not np.all(np.isfinite(control_signals)):
                raise RuntimeError("OSQP returned non-finite control values")

            # Ensure we never propagate numerically unstable controls.
            umin = self.input_constraints['umin']
            umax = self.input_constraints['umax']
            control_signals[0::2] = np.clip(control_signals[0::2], umin[0], umax[0])
            control_signals[1::2] = np.clip(control_signals[1::2], umin[1], umax[1])

            control_signals[1::2] = np.arctan(control_signals[1::2] *
                                              self.model.length)
            v = control_signals[0]
            delta = control_signals[1]

            if not np.isfinite(v) or not np.isfinite(delta):
                raise RuntimeError("MPC command contains non-finite values")

            # Update control signals
            self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[:(self.N+1)*nx], (self.N+1, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x)

            # Get current control signal
            u = np.array([v, delta])
            self.last_applied_control = np.array(u, dtype=float)

            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0
            self._log_recovery_if_needed()

        except Exception as exc:

            self._log_fallback(f"invalid solution ({exc})")
            u = self._get_fallback_control()

            # increase infeasibility counter
            self.infeasibility_counter += 1

        if self.infeasibility_counter >= (self.N - 1):
            print('No valid control signal computed for multiple steps; keeping last valid fallback control.')
            u = self._get_fallback_control()

        return u

    def update_prediction(self, spatial_state_prediction):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, self.N):
            # Get associated waypoint
            associated_waypoint = self.model.reference_path.\
                get_waypoint(self.model.wp_id+n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred

    def show_prediction(self):
        """
        Display predicted car trajectory in current axis.
        """

        if self.current_prediction is not None:
            plt.scatter(self.current_prediction[0], self.current_prediction[1],
                    c=PREDICTION, s=30)
