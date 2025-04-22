import numpy as np
from scipy.optimize import minimize
from generate_phi import *
import math
from self_tuning import *
from _PARAMETERS import *

# Common parameters
learning_rate = 0.05  # Learning rate
u_max = 5.0  # Control bound
T = 1000
N = 5  # Prediction horizon
k = 5  # Delay for DCL



# Dynamics functions
def chaotic_map_dynamics(x, u, phi):
    """Dynamics for a chaotic map system."""
    a = 3.8
    b = 0.1
    return a * x * (1 - x) + b * u + phi * np.sin(x)

def robotic_arm_dynamics(x, u, phi):
    """Robotic arm angle dynamics."""
    alpha = 0.5
    beta = 0.2
    return x + alpha * np.sin(x) + beta * u * np.exp(-np.abs(x)) + phi * x**3

# Cost functions
def chaotic_map_cost(x, u):
    """Cost for chaotic map."""
    mu = 0.1
    x_star = (a - 1) / a  # Fixed point ~ 0.7368
    return (x - x_star)**2 + mu * u**2

def robotic_arm_cost(x, u):
    """Cost for robotic arm: penalize deviation and torque."""

    gamma = 0.1
    return x**2 + gamma * u**2

def linear_dynamics(x, u, phi):
    """Linear dynamics."""
    return A @ x + B @ u + phi

def quadratic_cost(x, u):
    """Cost for robotic arm: penalize deviation and torque."""
    return x.T @ Q @ x + u.T @ R @ u





# MPC solver
# def mpc_solve(x_t, phi_seq, N, system_type):
#     """Solve MPC optimization for the given system type."""
#     def cost(U):
#         x = x_t
#         total_cost = 0
#         for k in range(N):
#             u = U[k]
#             # total_cost += (x - x_star) ** 2 + mu * u ** 2
#             # x = dynamics(x, u, phi_seq[k])
#             if system_type == 'chaotic_map':
#                 total_cost += chaotic_map_cost(x, u)
#                 x = chaotic_map_dynamics(x, u, phi_seq[k])
#             elif system_type == 'robotic_arm':
#                 total_cost += robotic_arm_cost(x, u)
#                 x = robotic_arm_dynamics(x, u, phi_seq[k])
#         return total_cost
#
#     bounds = [(-u_max, u_max)] * N
#     U0 = np.zeros(N)  # Initial guess
#     res = minimize(cost, U0, bounds=bounds, method='trust-constr')
#     return res.x[0]  # First control action

def mpc_solve(x_t, phi_seq, N, system_type):
    def cost(U):
        x = x_t
        __total_cost = 0
        for k in range(N):
            u = U[k]
            if system_type == 'chaotic_map':
                __total_cost +=  chaotic_map_cost(x, u) / N  # Scaled cost
                x = np.clip(chaotic_map_dynamics(x, u, phi_seq[k]), -10, 10)
                # print("x is:" + str(x))
                # print("u is:" + str(u))
            elif system_type == 'robotic_arm':
                __total_cost += robotic_arm_cost(x, u) / N
                x = np.clip(robotic_arm_dynamics(x, u, phi_seq[k]), -10 * np.pi, 10 * np.pi)
        return __total_cost

    def LQR_cost(U):
        # Reshape U to (N, 2) since u is in R^2
        U = U.reshape(N, 2)
        x = x_t
        __total_cost = 0
        for k in range(N):
            u = U[k]
            if system_type == 'linear':
                __total_cost += quadratic_cost(x, u) / N
                x = np.clip(linear_dynamics(x, u, phi_seq[k]), -10, 10)
        return __total_cost

    # Define control dimension based on system type
    if system_type == 'linear':
        m = 2  # u in R^2 for linear system
    else:
        m = 1  # Scalar u for other systems

    # Set bounds for each component of U (N * m elements)
    bounds = [(-u_max, u_max)] * (N * m)

    # Initial guess: zeros with length N * m
    U0 = np.zeros(N * m)

    # Optimize using SLSQP
    if system_type == 'linear':
        res = minimize(LQR_cost, U0, bounds=bounds, method='SLSQP')
    else:
        res = minimize(cost, U0, bounds=bounds, method='SLSQP')

    # Return the first control action (first m elements of res.x)
    return res.x[:m]

    # U0 = np.zeros(N)  # Or a better guess
    # res = minimize(cost, U0, bounds=bounds, method='SLSQP')
    # return res.x[0]  # First control action

# DCL algorithm

def dcl(phi_true_past, phi_pred_past, kappa_past, lambda_prev, learning_rate):
    """
    Delayed Confidence Learning update based on step-wise prediction errors.

    Parameters:
    - phi_true_past: Array of true values phi_star_tau for tau = t - k to t - 1
                     Shape: (k,) if scalar, or (k, dim) if vector
    - phi_pred_past: Array of predictions phi_{tau | t - k} for tau = t - k to t - 1
                     Shape: same as phi_true_past
    - kappa_past: Array of nominal values kappa_{tau | t - k} for tau = t - k to t - 1
                  Shape: same as phi_true_past
    - lambda_prev: Previous lambda value (lambda_{t - k})
    - beta: Learning rate (default 0.01)

    Returns:
    - lambda_new: Updated lambda_t, constrained to [0, 1]
    """
    # Compute prediction errors
    if len(phi_true_past) == 1:  # Scalar case
        eps = np.abs(phi_pred_past - phi_true_past)  # ε_{τ|t} = |phi_star - phi_pred|
        eps_bar = np.abs(kappa_past - phi_true_past)  # ε̅_{τ|t} = |phi_star - kappa|
    else:  # Vector case
        eps = np.linalg.norm(phi_pred_past - phi_true_past)  # Euclidean norm
        eps_bar = np.linalg.norm(kappa_past - phi_true_past)  # Euclidean norm

    # Compute gradient of the loss function ξ_{t - k}(λ)
    # ξ = Σ [λ² ε² + (1 - λ)² ε̅²]
    # ∂ξ/∂λ = Σ [2λ ε² - 2(1 - λ) ε̅²]
    grad = np.sum(2 * lambda_prev * eps ** 2 - 2 * (1 - lambda_prev) * eps_bar ** 2)

    # Update lambda using gradient descent
    lambda_new = lambda_prev - learning_rate * grad

    # Project lambda onto [0, 1]
    return np.clip(lambda_new, 0, 1)

def run_dynamics(Noise_mu, sigma_eta, SYSTEM_TYPE, METHOD_TYPE):

    # Simulation

    if SYSTEM_TYPE == 'linear':
        x_t = np.zeros(4)
    else:
        x_t = 0.5  # Initial state
    x_history = [x_t]

    u_history = []
    lambda_history = []

    # Histories for DCL
    lambda_values = [0.5] * k  # Initialize lambda for t < k
    np.random.seed(42)  # For reproducibility

    # Histories for STC

    eta_true_past = []
    eta_pred_past = []

    # Generate true phi_t
    phi_true_history = generate_phi(T, N, SYSTEM_TYPE)

    # Store predictions and lambda
    phi_pred_history = []
    total_cost = 0

    for t in range(T):

        print("time is " + str(t))

        # Generate predictions with noise
        phi_pred = []
        for i in range(phi_true_history.shape[1]):
            phi_pred.append(phi_true_history[t: t+k][:, i] + np.random.normal(Noise_mu, sigma_eta, k))

        if SYSTEM_TYPE == 'linear':
            phi_pred = np.array(phi_pred).T

        phi_pred_history.append(phi_pred)

        # Nominal values

        if SYSTEM_TYPE == 'linear':
            kappa = np.zeros((N, 4))
        else:
            kappa = np.zeros(N)

        # MPC solutions
        u_phi = mpc_solve(x_t, phi_pred[:N], N, SYSTEM_TYPE)
        u_kappa = mpc_solve(x_t, kappa[:N], N, SYSTEM_TYPE)

        # Control action
        if METHOD_TYPE == 'lac':

            # Confidence learning
            if t <= k:
                lambda_t = 0.5
            else:
                # Get past data for DCL
                idx = t - k
                phi_true_past = np.array(phi_true_history[idx:t])
                phi_pred_past = np.array(phi_pred_history[idx][0:len(phi_true_past)])
                if SYSTEM_TYPE == 'linear':
                    kappa_past = np.zeros((len(phi_true_past), 4))
                else:
                    kappa_past = np.zeros(len(phi_true_past))
                lambda_prev = lambda_values[-k]
                lambda_t = dcl(phi_true_past, phi_pred_past, kappa_past, lambda_prev, learning_rate)
            u_t = lambda_t * u_phi + (1 - lambda_t) * u_kappa
            lambda_values.append(lambda_t)
            lambda_history.append(lambda_t)

        elif METHOD_TYPE == 'self-tuning' and SYSTEM_TYPE == 'linear':
            phi_true_past = np.array(phi_true_history[t])
            phi_pred_past = []
            if t > 0:
                for i in range(min(t, N)):
                    phi_pred_past.append(phi_pred_history[t - i - 1][i])
            phi_pred_past = np.array(phi_pred_past)
            lambda_t, eta_true_past, eta_pred_past = self_tuning_ftl(N, t, phi_true_past, phi_pred_past, P, F, H, eta_true_past, eta_pred_past)
            u_t = lambda_t * u_phi + (1 - lambda_t) * u_kappa
            lambda_history.append(lambda_t)

        elif METHOD_TYPE == '1-mpc':
            u_t = u_phi
        elif METHOD_TYPE == '0-mpc':
            u_t = u_kappa

        u_history.append(u_t)

        # Update state
        if SYSTEM_TYPE == 'chaotic_map':
            x_t = chaotic_map_dynamics(x_t, u_t, phi_true_history[t])
            total_cost += chaotic_map_cost(x_t, u_t)  # Scaled cost
        elif SYSTEM_TYPE == 'robotic_arm':
            x_t = robotic_arm_dynamics(x_t, u_t, phi_true_history[t])
            total_cost += robotic_arm_cost(x_t, u_t)  # Scaled cost
        elif SYSTEM_TYPE == 'linear':
            x_t = linear_dynamics(x_t, u_t, phi_true_history[t])
            total_cost += quadratic_cost(x_t, u_t) / N # Scaled cost
        x_history.append(x_t)

    return total_cost, x_history, u_history, lambda_history, phi_pred_history, phi_true_history
