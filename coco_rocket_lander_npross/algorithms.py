# vim: ts=4 sw=4 noet spell tw=80:

import time
import logging
log = logging.getLogger(__name__)

import numpy as np
import cvxpy as cp
import scipy

from coco_rocket_lander.algs import PID_Benchmark
from coco_rocket_lander.env import SystemModel


class Algorithm:
    def setup(self, env, landing_pos):
        """ Set up the algorithm, if necessary """
        pass

    def run(self, x):
        """ Run the algorithm """
        raise NotImplementedError

    @classmethod
    @property
    def name(cls):
        return cls.__qualname__

    def make_picklable(self):
        raise NotImplementedError


class PID(Algorithm):
    """
    Reference 'benchmark' PID algorithm
    """
    def __init__(self, sample_time=None):
        self.sample_time = sample_time

        # Default values from jupyter notebook
        self.engine = [10, 0, 10]
        self.vector = [0.085, 0.001, 10.55]
        self.side = [5, 0, 6]

    def setup(self, env, landing_pos, engine=None, vector=None, side=None):
        if engine:
            self.engine = engine
        if vector:
            self.vector = vector
        if side:
            self.side = side

        self.lpos = landing_pos
        self.pid = PID_Benchmark(self.engine, self.vector, self.side)

    def run(self, x):
        log.debug(f"State x={x}")
        # Remove position offset
        start = time.time()
        x[0] -= self.lpos[0]
        x[1] -= self.lpos[1]
        action = np.array(self.pid.pid_algorithm(x))
        solvetime = (time.time() - start)
        log.debug(f"Computation took {solvetime:.2e}, action={action}")
        return action, solvetime

    def make_picklable(self):
        return self


class ClassicMPC(Algorithm):
    """
    Classic MPC
    """
    def __init__(self, sample_time=.25, time_horizon=10):
        self.sample_time = sample_time
        self.time_horizon = time_horizon
        self.horizon = int(time_horizon / self.sample_time)

    def setup(self, env, landing_pos):
        self.setup_model(env)
        self.setup_constraints(landing_pos)
        self.setup_problem(landing_pos)

    def setup_model(self, env):
        # Get model parameters from environment
        model = SystemModel(env)
        self.scale_u = np.diag([
            1 / env.cfg.main_engine_thrust,
            1 / env.cfg.side_engine_thrust,
            1 / env.cfg.max_nozzle_angle
        ])

        # Linearization (stationary) points
        self.zs = np.zeros([6,])
        self.us = np.array([model.mass * model.gravity, 0, 0])
        self.scaled_us = self.scale_u @ self.us

        # Get linearised model of the system
        model.calculate_linear_system_matrices(self.zs, self.us)
        model.discretize_system_matrices(sample_time=self.sample_time)

        # Store to use later
        self.A, self.B = model.get_discrete_linear_system_matrices()
        self.nz, self.nu = self.B.shape

    def setup_constraints(self, landing_pos):
        # State constraints
        self.Gx = np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, -1, 0],
            [0, -1, 0, 0, 0, 0]])
        self.gx = np.array([.6, .6, -landing_pos[1] + 8])
        log.debug(f"State constraints: gx=\n{self.gx},\nGx=\n{self.Gx} ")

        # Input constraints
        self.Gu = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        self.gu = np.array([1, 1, 1, 0, 1, 1])
        log.debug(f"Input constraints: gu=\n{self.gu},\nGu=\n{self.Gu} ")

    def setup_problem(self, landing_pos):
        # Terminal state
        self.zf = np.hstack([landing_pos[:2], np.zeros((4,))]) - self.zs
        log.debug(f"Terminal state zf={self.zf}")

       # Tuning parameters for MPC
        self.Q = np.diag([10, 1, 10, 1, 500, 200])
        self.S = np.eye(self.nz) * 100
        self.R = np.diag([1, 10, 10])
        self.v = 1

        # Set up CVXPY for MPC problem
        self.z0 = cp.Parameter((self.nz,))
        self.z = cp.Variable((self.nz, self.horizon+1))
        self.u = cp.Variable((self.nu, self.horizon))

        # Slack variables (relaxation)
        self.s = cp.Variable((self.gx.shape[0], self.horizon+1))

        # Add initial condition
        constraints = [self.z[:, 0] == self.z0]

        # Add cost-to-go
        cost = 0
        for k in range(self.horizon):
            cost += cp.quad_form(self.z[:, k], self.Q)
            cost += self.v * cp.norm(self.s[:, k], 1)
            cost += cp.quad_form(self.u[:, k], self.R)
            constraints += [
                self.z[:, k+1] == self.A @ self.z[:, k] + self.B @ self.u[:, k],
                self.Gx @ self.z[:, k] <= self.gx + self.s[:, k] - self.Gx @ self.zs,
                self.Gu @ self.u[:, k] <= self.gu - self.Gu @ self.scaled_us
            ]

        # Add terminal cost and constraint
        cost += cp.quad_form(self.z[:, self.horizon], self.S)
        cost += self.v * cp.norm(self.s[:, self.horizon], 1)
        constraints.append(self.z[:, self.horizon] == np.zeros((self.nz,)))

        # Create optimization problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def run(self, z):
        # Stop if there is nothing to do
        if z[6] and z[7]:
            return np.zeros(self.nu), 0

        # Give relative coordinate
        self.z0.value = z[:-2] - self.zf

        log.debug(f"Solving MPC optimization with z0={self.z0.value}")
        self.problem.solve()

        if self.problem.status in ["infeasible", "unbounded"]:
            log.error(f"Problem cannot be solved, it is {self.problem.status}")
            raise RuntimeError("Impossible optimization problem")

        action = self.scaled_us + self.u[:, 0].value
        solvetime = self.problem.solver_stats.solve_time
        log.debug(f"Solver took {solvetime:.4e}, result={action}")

        return action, solvetime


class OnlineMPC(Algorithm):
    """
    Relaxed linear MPC over linearised dynamics.
    """
    def __init__(self, sample_time=.25, time_horizon=10):
        self.sample_time = sample_time
        self.time_horizon = time_horizon
        self.horizon = int(time_horizon / self.sample_time)

    def setup(self, env, landing_pos):
        horizon = self.horizon
        log.info(f"Setup MPC with horizon={horizon}")

        # Setup model and landing position
        model = SystemModel(env)
        self.scale_u = np.diag([
            1 / env.cfg.main_engine_thrust,
            1 / env.cfg.side_engine_thrust,
            1 / env.cfg.max_nozzle_angle
        ])

        # Linearization (stationary) points
        self.xs = np.zeros([6,])
        self.us = np.array([model.mass * model.gravity, 0, 0])
        self.scaled_us = self.scale_u @ self.us

        # Terminal state
        self.xf = np.hstack([landing_pos[:2], np.zeros((4,))]) - self.xs
        log.debug(f"Terminal state xf={self.xf}")

        # Get linearised model of the system
        model.calculate_linear_system_matrices(self.xs, self.us)
        model.discretize_system_matrices(sample_time=self.sample_time)

        model.B = model.B @ self.scale_u
        self.A, self.B = model.get_discrete_linear_system_matrices()
        self.nx, self.nu = self.B.shape

        # Parameters for MPC
        Q = [
            np.diag([
                1 + 10*k/horizon, 1,
                1 + 10*k/horizon, 2,
                100 + 100*k/horizon, 200]) 
            for k in range(horizon)
        ]
        S = np.eye(self.nx) * 100
        R = np.diag([1, 10, 10])

        # State constraints
        self.Gx = np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, -1, 0],
            [0, -1, 0, 0, 0, 0]])
        self.gx = np.array([.6, .6, -landing_pos[1] + 8])
        log.debug(f"State constraints: gx=\n{self.gx},\nGx=\n{self.Gx} ")

        # Input constraints
        self.Gu = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        self.gu = np.array([1, 1, 1, 0, 1, 1])
        log.debug(f"Input constraints: gu=\n{self.gu},\nGu=\n{self.Gu} ")

        # Set up CVXPY for MPC problem
        self.x0 = cp.Parameter((self.nx,))
        self.x = cp.Variable((self.nx, horizon+1))
        self.u = cp.Variable((self.nu, horizon))

        # Slack variables (relaxation)
        self.s = cp.Variable((self.gx.shape[0], horizon+1))

        # Add initial condition
        constraints = [self.x[:, 0] == self.x0]

        # Add cost-to-go
        cost = 0
        for k in range(horizon):
            cost += cp.quad_form(self.x[:, k], Q[k])
            cost += 1e3 * cp.norm(self.s[:, k], 1)
            cost += cp.quad_form(self.u[:, k], R)
            constraints += [
                self.x[:, k+1] == self.A @ self.x[:, k] + self.B @ self.u[:, k],
                self.Gx @ self.x[:, k] <= self.gx + self.s[:, k] - self.Gx @ self.xs,
                self.Gu @ self.u[:, k] <= self.gu - self.Gu @ self.scaled_us
            ]

        # Add terminal cost and constraint
        cost += cp.quad_form(self.x[:, horizon], S)
        cost += cp.norm(self.s[:, horizon], 1)
        constraints.append(self.x[:, horizon] == np.zeros((self.nx,)))

        # Create optimization problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def run(self, x):
        # Stop if there is nothing to do
        if x[6] and x[7]:
            return np.zeros(self.nu), 0

        # Give relative coordinate
        self.x0.value = (x[:-2].reshape(self.xf.shape) - self.xf).reshape(self.x0.shape)

        log.debug(f"Solving MPC optimization with x0={self.x0.value}")
        self.problem.solve()

        # Slower solver
        # self.problem.solve(warm_start=True, solver=cp.SCS, eps=1e-4)

        if self.problem.status in ["infeasible", "unbounded"]:
            log.error(f"Problem cannot be solved, it is {self.problem.status}")
            raise RuntimeError("Impossible optimization problem")

        action = self.scaled_us + self.u[:, 0].value
        solvetime = self.problem.solver_stats.solve_time
        log.debug(f"Solver took {solvetime:.4e}, "
            f"result={action}")

        return action, solvetime

    def make_picklable(self):
        self.problem = None
        return self


class QLearning(Algorithm):
    def setup(self, env, landing_pos, weights=None):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
