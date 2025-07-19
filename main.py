import numpy as np
from scipy.optimize import minimize
from generate_phi import *
import math
from self_tuning import *
from _PARAMETERS import *
from dynamics import *

# Common parameters
learning_rate = 0.05 # Learning rate
if SYSTEM_TYPE == 'robotic_arm':
    u_max = 1000.0
else:
    u_max = 10.0  # Control bound
T = 200

if SYSTEM_TYPE == 'robotic_arm':
    N = 5  # Prediction horizon
    k = 5 
else:
    N = 5  # Prediction horizon
    k = 5 
 # Delay for DCL


def sample_on_sphere(noise_mu, sigma, k):
    """
    Uniformly sample a 4-dimensional vector with fixed L2-norm L.

    Parameters:
    - L: The desired L2-norm (radius of the sphere).

    Returns:
    - A 4-dimensional vector with L2-norm L, uniformly sampled on the sphere.
    """
    # Generate a 4-dimensional vector from standard normal distribution
    # X = np.random.randn(k)
    X = np.random.normal(0, sigma, k)
    # Compute the L2-norm of X
    norm_X = np.linalg.norm(X)

    # Normalize X and scale to the desired norm L
    Y = noise_mu * (X / norm_X)
    # print(np.linalg.norm(Y))
    return Y


def mpc_solve(x_t, phi_seq, N, system_type):
    def cost(U):
        x = x_t
        __total_cost = 0
        for k in range(N):
            u = U[k]
            if system_type == 'robotic_arm':
                __total_cost += robotic_arm_cost(x, u) / N
                x = np.clip(robotic_arm_dynamics(x, u, phi_seq[k]), -100, 100)
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
    elif system_type == 'robotic_arm':
        m = 1  # Scalar u for other systems

    # Set bounds for each component of U (N * m elements)
    bounds = [(-u_max, u_max)] * (N * m)

    # Initial guess: zeros with length N * m
    U0 = np.zeros(N * m)

    # Define constraints for the optimizer
    constraints = []

    if system_type == 'robotic_arm':
        def state_constraint(U):
            # U is a flat array of length N
            x = x_t
            cons = []
            for k in range(N):
                u = U[k]
                x = robotic_arm_dynamics(x, u, phi_seq[k])
                cons.append(x - 0.2)   # x <= 2 --> x - 2 <= 0
                cons.append(-x - 0.2)  # x >= -2 --> -x - 2 <= 0
            return np.array(cons)
        constraints = [{'type': 'ineq', 'fun': state_constraint}]

    # Optimize using SLSQP
    if system_type == 'linear':
        res = minimize(LQR_cost, U0, bounds=bounds, method='SLSQP')
    elif system_type == 'robotic_arm':
        res = minimize(cost, U0, bounds=bounds, constraints=constraints, method='SLSQP')
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
    # if len(phi_true_past) == 1:  # Scalar case
    eps = phi_pred_past - phi_true_past  # ε_{τ|t} = |phi_star - phi_pred|
    eps_bar = kappa_past - phi_true_past  # ε̅_{τ|t} = |phi_star - kappa|

    # else:  # Vector case
    #     eps = np.linalg.norm(phi_pred_past - phi_true_past)  # Euclidean norm
    #     eps_bar = np.linalg.norm(kappa_past - phi_true_past)  # Euclidean norm

    # Compute gradient of the loss function ξ_{t - k}(λ)
    # ξ = Σ [λ² ε² + (1 - λ)² ε̅²]
    # ∂ξ/∂λ = Σ [2λ ε² - 2(1 - λ) ε̅²]
    grad = np.sum(2 * lambda_prev * np.linalg.norm(eps) ** 2 - 2 * (1 - lambda_prev) * np.linalg.norm(eps_bar) ** 2)

    # grad = 0
    # for tau in range(len(eps)):
    #     grad += (lambda_prev * eps[tau] + (1-lambda_prev) * eps_bar[tau]).T @ (eps[tau] - eps_bar[tau])

        # grad_l2 = np.sum(eps - eps_bar)
        # grad += (rho ** (2 * tau)) * (lambda_prev * eps[tau] + (1-lambda_prev) * eps_bar[tau]).T @ (eps[tau] - eps_bar[tau])

    # Update lambda using gradient descent
    lambda_new = lambda_prev - learning_rate * np.clip(grad, -100, 100)

    # Project lambda onto [0, 1]
    return np.clip(lambda_new, 0, 1)


def run_dynamics(noise_mu, sigma, SYSTEM_TYPE, METHOD_TYPE, number_tests):

    all_total_cost = []
    all_x_history = {}
    all_u_history = {}
    all_lambda_history = {}
    all_phi_pred_history = {}
    all_phi_true_history = {}

    # Simulation

    for num in range(number_tests):

        if SYSTEM_TYPE == 'linear':
            x_t = np.zeros(4)
        else:
            x_t = 0.5  # Initial state
        x_history = [x_t]

        u_history = []
        lambda_history = []

        # Histories for DCL
        lambda_values = [0.5] * k  # Initialize lambda for t < k
        np.random.seed(num+6)  # For reproducibility

        # Histories for STC

        eta_true_past = []
        eta_pred_past = []

        # Generate true phi_t
        phi_true_history = generate_phi(T, N, SYSTEM_TYPE)

        # Store predictions and lambda
        phi_pred_history = []
        total_cost = 0

        for t in range(T):

            # print("time is " + str(t))

            # Generate predictions with noise
            phi_pred = []

            if with_attacks is True and SYSTEM_TYPE == 'linear':
                attack_time = T // 3
                terminal_time = 2 * (T // 3) 
                attack_interval = 5
                if t < attack_time:
                    for i in range(phi_true_history.shape[1]):
                        phi_pred.append(phi_true_history[t: t+k][:, i])
                elif t % attack_interval <= 1 and t < terminal_time:
                    for i in range(phi_true_history.shape[1]):
                        phi_pred.append(phi_true_history[t: t+k][:, i] + sample_on_sphere(4, 0.5, k))
                else:
                    for i in range(phi_true_history.shape[1]):
                        phi_pred.append(phi_true_history[t: t+k][:, i])
            elif SYSTEM_TYPE == 'linear':
                for i in range(phi_true_history.shape[1]):
                    phi_pred.append(phi_true_history[t: t+k][:, i] + sample_on_sphere(noise_mu, sigma, k))
            elif with_attacks is True and SYSTEM_TYPE == 'robotic_arm':
                attack_time = T // 3
                terminal_time = 2 * (T // 3) 
                attack_interval = 5
                if t < attack_time:
                    phi_pred.append(phi_true_history[t: t+k])
                elif t % attack_interval <= 1 and t < terminal_time:
                    phi_pred.append(phi_true_history[t: t+k] + sample_on_sphere(4, 0.5, k))
                else:
                    phi_pred.append(phi_true_history[t: t+k])


            # if SYSTEM_TYPE == 'linear':
            phi_pred = np.array(phi_pred).T

            phi_pred_history.append(phi_pred)

            # Nominal values

            if SYSTEM_TYPE == 'linear':
                kappa = np.zeros((N, 4))
            elif SYSTEM_TYPE == 'robotic_arm':
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

            elif METHOD_TYPE == 'self-tuning':
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

            if SYSTEM_TYPE == 'robotic_arm':
                x_t = robotic_arm_dynamics(x_t, u_t, phi_true_history[t])
                total_cost += robotic_arm_cost(x_t, u_t) / N  # Scaled cost
            elif SYSTEM_TYPE == 'linear':
                x_t = linear_dynamics(x_t, u_t, phi_true_history[t])
                total_cost += quadratic_cost(x_t, u_t) / N # Scaled cost

            x_history.append(x_t)

        all_total_cost.append(total_cost)
        all_x_history[num] = x_history
        all_u_history[num] = u_history
        all_lambda_history[num] = lambda_history
        all_phi_pred_history[num] = phi_pred_history
        all_phi_true_history[num] = phi_true_history

    return all_total_cost, all_x_history, all_u_history, all_lambda_history, all_phi_pred_history, all_phi_true_history
