from main import *
import matplotlib.pyplot as plt

# System parameters
SYSTEM_TYPE = 'linear'  # Options: 'chaotic_map' or 'robotic_arm' or 'linear'
sigma_eta = 0.5
noise_mu = 0
total_cost = {}

x_history = {}
u_history = {}
lambda_history = {}
phi_pred_history = {}
phi_true_history = {}
#
#
# def plot_control_costs(costs_method1, costs_method2, costs_method3, time_steps):
#     """
#     Plots instantaneous costs for three control methods over time.
#
#     Parameters:
#     - costs_method1: Array or list of costs for Method 1 over time.
#     - costs_method2: Array or list of costs for Method 2 over time.
#     - costs_method3: Array or list of costs for Method 3 over time.
#     - time_steps: Array or list of time steps (e.g., np.arange(T)).
#
#     The function generates a plot of instantaneous costs for each method over time.
#     Total costs are displayed in the legend for each method.
#     """
#     # Convert inputs to numpy arrays for consistency
#     costs_mean1 = np.array(np.mean(costs_method1, axis=1))
#     std_dev1 = np.std(costs_method1, axis=1)
#     costs_mean2 = np.array(np.mean(costs_method2, axis=1))
#     std_dev2 = np.std(costs_method2, axis=1)
#     costs_mean3 = np.array(np.mean(costs_method3, axis=1))
#     std_dev3 = np.std(costs_method3, axis=1)
#     time_steps = np.array(np.mean(time_steps))
#
#     # Check that all cost arrays have the same length as time_steps
#     if not (len(costs_method1) == len(costs_method2) == len(costs_method3) == len(time_steps)):
#         raise ValueError("All cost arrays and time_steps must have the same length.")
#
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # Plot each method's costs with distinct styles
#     ax.plot(time_steps, costs_method1, label=f'Method 1', color='blue', linestyle='-')
#     plt.fill_between(range(len(time_steps)), costs_mean1 - std_dev1/2, costs_mean1 + std_dev1/2, alpha=0.3)
#     ax.plot(time_steps, costs_method2, label=f'Method 2', color='green', linestyle='--')
#     plt.fill_between(range(len(time_steps)), costs_mean2 - std_dev2/2, costs_mean2 + std_dev2/2, alpha=0.3)
#     ax.plot(time_steps, costs_method3, label=f'Method 3', color='red', linestyle='-.')
#     plt.fill_between(range(len(time_steps)), costs_mean3 - std_dev3/2, costs_mean3 + std_dev3/2, alpha=0.3)
#
#     # Set title, labels, legend, and grid
#     ax.set_title('Comparison of Control Method Costs Over Time')
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Cost')
#     ax.legend(loc='upper right')
#     ax.grid(True)
#     # plt.ylim((0.35, 0.5))
#
#     # Adjust layout and display the plot
#     plt.tight_layout()
#     plt.savefig('results', bbox_inches='tight')
#     plt.show()


def plot_control_costs(costs_method1, costs_method2, costs_method3, costs_method4, time_steps):
    """
    Plots instantaneous costs for three control methods over time.

    Parameters:
    - costs_method1: Array or list of costs for Method 1 over time.
    - costs_method2: Array or list of costs for Method 2 over time.
    - costs_method3: Array or list of costs for Method 3 over time.
    - costs_method4: Array or list of costs for Method 3 over time.
    - time_steps: Array or list of time steps (e.g., np.arange(T)).

    The function generates a plot of instantaneous costs for each method over time.
    Total costs are displayed in the legend for each method.
    """

    plt.rcParams.update({'font.size': 22})
    plt.rcParams['axes.linewidth'] = 2  # set the value globally

    # Convert inputs to numpy arrays for consistency
    costs_mean1 = np.array(np.mean(costs_method1, axis=1))
    std_dev1 = np.std(costs_method1, axis=1)
    costs_mean2 = np.array(np.mean(costs_method2, axis=1))
    std_dev2 = np.std(costs_method2, axis=1)
    costs_mean3 = np.array(np.mean(costs_method3, axis=1))
    std_dev3 = np.std(costs_method3, axis=1)
    costs_mean4 = np.array(np.mean(costs_method4, axis=1))
    std_dev4 = np.std(costs_method4, axis=1)
    # time_steps = np.array(np.mean(time_steps))

    # Check that all cost arrays have the same length as time_steps
    if not (len(costs_mean1) == len(costs_mean2) == len(costs_mean3) == len(costs_mean4) == len(time_steps)):
        raise ValueError("All cost arrays and time_steps must have the same length.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each method's costs with distinct styles
    ax.plot(time_steps, costs_mean1, label=f'LAC', color='blue', linestyle='-', linewidth=2)
    plt.fill_between(range(len(time_steps)), costs_mean1 - std_dev1/2, costs_mean1 + std_dev1/2, alpha=0.4)
    ax.plot(time_steps, costs_mean2, label=f'MPC', color='green', linestyle='--', linewidth=2)
    plt.fill_between(range(len(time_steps)), costs_mean2 - std_dev2/2, costs_mean2 + std_dev2/2, alpha=0.4)
    ax.plot(time_steps, costs_mean3, label=f'LQR', color='red', linestyle='-.', linewidth=3)
    plt.fill_between(range(len(time_steps)), costs_mean3 - std_dev3/2, costs_mean3 + std_dev3/2, alpha=0.4)
    ax.plot(time_steps, costs_mean4, label=f'Self-Tuning', color='black', linestyle='-', linewidth=2)
    plt.fill_between(range(len(time_steps)), costs_mean4 - std_dev4/2, costs_mean4 + std_dev4/2, alpha=0.4)

    # Set title, labels, legend, and grid
    # ax.set_title('Comparison of Control Method Costs Over Time')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Cost')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.ylim((110, 200))

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('linear', bbox_inches='tight')
    plt.show()


total_cost['lac'] = []
total_cost['1-mpc'] = []
total_cost['0-mpc'] = []
total_cost['self-tuning'] = []

for mu in range(1, 50, 1):
    # noise_mu = mu/3
    noise_mu = mu/10
    total_cost_lac, x_history[noise_mu], u_history[noise_mu], lambda_history[noise_mu], phi_pred_history[noise_mu], phi_true_history[noise_mu] = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'lac', 5)
    total_cost_1mpc, x_history[noise_mu], u_history[noise_mu], lambda_history[noise_mu], phi_pred_history[noise_mu], phi_true_history[noise_mu] = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '1-mpc', 5)
    total_cost_0mpc, x_history[noise_mu], u_history[noise_mu], lambda_history[noise_mu], phi_pred_history[noise_mu], phi_true_history[noise_mu] = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '0-mpc', 5)
    total_cost_st, x_history[noise_mu], u_history[noise_mu], lambda_history[noise_mu], phi_pred_history[noise_mu], phi_true_history[noise_mu] = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'self-tuning', 5)
    total_cost['lac'].append(total_cost_lac)
    total_cost['1-mpc'].append(total_cost_1mpc)
    total_cost['0-mpc'].append(total_cost_0mpc)
    total_cost['self-tuning'].append(total_cost_st)

time_steps = np.arange(len(total_cost['lac']))
plot_control_costs(total_cost['lac'], total_cost['1-mpc'], total_cost['0-mpc'], total_cost['self-tuning'], time_steps)
# plot_control_costs(total_cost['lac'], total_cost['1-mpc'], total_cost['0-mpc'], time_steps)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(x_history, label='State x_t')
plt.axhline(x_star, color='r', linestyle='--', label='Reference x*')
plt.legend()
plt.title('State Trajectory')

# plt.subplot(3, 1, 2)
# plt.plot(u_history, label='Control u_t')
# plt.legend()
# plt.title('Control Input')

plt.subplot(3, 1, 2)
plt.plot(u_history, label='Control u_t')
plt.legend()
plt.title('Control Input')

plt.subplot(3, 1, 3)
plt.plot(lambda_history, label='Confidence lambda_t')
plt.legend()
plt.title('Learned Confidence')

plt.tight_layout()
plt.show()


# Visualize prediction error
# Collect predicted phi_t used in MPC
phi_pred_used = [phi_pred_history[t][0] for t in range(T)]

# Plot true phi_t vs predicted phi_t
plt.figure(figsize=(12, 6))
plt.plot(phi_true_history[:T], label='True $\phi_t$')
plt.plot(phi_pred_used, label='Predicted $\phi_t$ (used in MPC)')
plt.legend()
plt.title('True and Predicted $\phi_t$')
plt.xlabel('Time step $t$')
plt.ylabel('$\phi_t$')
plt.show()

# Plot prediction error
prediction_error = np.array(phi_true_history[:T]) - np.array(phi_pred_used)
plt.figure(figsize=(12, 6))
plt.plot(prediction_error, label='Prediction Error $\phi_t - \hat{\phi}_t$')
plt.axhline(0, color='r', linestyle='--')
plt.legend()
plt.title('Prediction Error for $\phi_t$')
plt.xlabel('Time step $t$')
plt.ylabel('Error')
plt.show()

# Compute and plot MAE for DCL
mae_history = []
for t in range(k, T):
    phi_true_past = phi_true_history[t - k:t]
    phi_pred_past = phi_pred_history[t - k]
    mae = np.mean(np.abs(phi_true_past - phi_pred_past))
    mae_history.append(mae)

plt.figure(figsize=(12, 6))
plt.plot(range(k, T), mae_history, label='MAE of predictions used in DCL')
plt.legend()
plt.title('Mean Absolute Error of Predictions for DCL')
plt.xlabel('Time step $t$')
plt.ylabel('MAE')
plt.show()

