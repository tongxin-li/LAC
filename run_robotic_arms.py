from main import *
import matplotlib.pyplot as plt
import numpy as np

# --- Simulation Setup ---
SYSTEM_TYPE = 'robotic_arm'
noise_mu = 0
sigma_eta = 0
T = 200  # Time horizon

# --- Run Dynamics for Each Method ---
print("Running LAC...")
cost_lac, x_lac, u_lac, lambda_lac, phi_pred, phi_true = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'lac', 1)
print("Running MPC...")
cost_mpc, x_mpc, u_mpc, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '1-mpc', 1)
print("Running LQR/Nominal...")
cost_lqr, x_lqr, u_lqr, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '0-mpc', 1)

# Helper function to compute instantaneous cost at each step
def get_inst_cost(x_hist, u_hist, system_type):
    costs = []
    cost_func_map = {
        'robotic_arm': robotic_arm_cost,
        'chaotic_map': chaotic_map_cost,
        'linear': quadratic_cost
    }
    cost_func = cost_func_map.get(system_type, quadratic_cost)
    num_steps = len(u_hist)
    for t in range(num_steps):
        costs.append(cost_func(x_hist[t + 1], u_hist[t]))
    return costs

# Calculate instantaneous costs for each method
inst_cost_lac = get_inst_cost(x_lac[0], u_lac[0], SYSTEM_TYPE)
inst_cost_mpc = get_inst_cost(x_mpc[0], u_mpc[0], SYSTEM_TYPE)
inst_cost_lqr = get_inst_cost(x_lqr[0], u_lqr[0], SYSTEM_TYPE)
time_steps = np.arange(T - 1)

# --- Plotting for Robotic Arm Case ---
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.fancybox': True,
    'legend.edgecolor': 'gray',
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

attack_time = T // 3
attack_end = 2 * T // 3

# --- Color Palette ---
color_lac = '#0072B2'
color_mpc = '#009E73'
color_lqr = '#E69F00'
color_st = '#666666'
color_attack = '#D55E00'
color_attack_region = '#D6C6E1'

# Helper for moving average smoothing
def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

# Prepare data
x_lac_arr = np.array(x_lac[0]).flatten()
x_mpc_arr = np.array(x_mpc[0]).flatten()
x_lqr_arr = np.array(x_lqr[0]).flatten()

u_lac_arr = np.array(u_lac[0]).flatten()
u_mpc_arr = np.array(u_mpc[0]).flatten()
u_lqr_arr = np.array(u_lqr[0]).flatten()

window = 5
inst_cost_lac_smooth = moving_average(inst_cost_lac, window)
inst_cost_mpc_smooth = moving_average(inst_cost_mpc, window)
inst_cost_lqr_smooth = moving_average(inst_cost_lqr, window)

# --- New Plotting Layout: Use gridspec for custom height ratios ---
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 13))
gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.2, 1.2, 0.5], hspace=0.32)

# 1. State (x) over time
ax0 = fig.add_subplot(gs[0])
ax0.plot(np.arange(len(x_lac_arr)), x_lac_arr, label='LAC $x$', color=color_lac)
ax0.plot(np.arange(len(x_mpc_arr)), x_mpc_arr, label='MPC $x$', color=color_mpc)
ax0.plot(np.arange(len(x_lqr_arr)), x_lqr_arr, label='LQR $x$', color=color_lqr)
ax0.axvspan(attack_time, attack_end, color=color_attack_region, alpha=0.35, zorder=1)
ax0.axvline(x=attack_time, color=color_attack, linestyle='--', linewidth=2.0, zorder=2)
ax0.set_ylabel('State $x$')
ax0.set_title('Robotic Arm State')
ax0.legend(ncol=1, fontsize=13, frameon=True, facecolor='white', edgecolor='gray')
ax0.grid(True, linestyle='--', alpha=0.8)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_xlim([0, T-1])

# 2. Control Input (u) over time
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.plot(np.arange(len(u_lac_arr)), u_lac_arr, label='LAC $u$', color=color_lac)
ax1.plot(np.arange(len(u_mpc_arr)), u_mpc_arr, label='MPC $u$', color=color_mpc)
ax1.plot(np.arange(len(u_lqr_arr)), u_lqr_arr, label='LQR $u$', color=color_lqr)
ax1.axvspan(attack_time, attack_end, color=color_attack_region, alpha=0.35, zorder=1)
ax1.axvline(x=attack_time, color=color_attack, linestyle='--', linewidth=2.0, zorder=2)
ax1.set_ylabel('Control Input $u$')
ax1.set_title('Robotic Arm Control Input')
ax1.grid(True, linestyle='--', alpha=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim([0, T-1])

# 3. Instantaneous Cost
ax2 = fig.add_subplot(gs[2], sharex=ax0)
min_len = min(len(time_steps), len(inst_cost_lac_smooth), len(inst_cost_mpc_smooth), len(inst_cost_lqr_smooth))
ax2.plot(time_steps[:min_len], inst_cost_lac_smooth[:min_len], label='LAC', color=color_lac)
ax2.plot(time_steps[:min_len], inst_cost_mpc_smooth[:min_len], label='MPC', color=color_mpc)
ax2.plot(time_steps[:min_len], inst_cost_lqr_smooth[:min_len], label='LQR', color=color_lqr)
ax2.axvspan(attack_time, attack_end, color=color_attack_region, alpha=0.35, zorder=1)
ax2.axvline(x=attack_time, color=color_attack, linestyle='--', linewidth=2.0, zorder=2)
ax2.set_ylabel('Instantaneous Cost')
ax2.set_title('Controller Performance')
ax2.grid(True, linestyle='--', alpha=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim([0, T-1])

# Annotate total costs
total_cost_lac = np.sum(cost_lac)
total_cost_mpc = np.sum(cost_mpc)
total_cost_lqr = np.sum(cost_lqr)
cost_text = (
    f"LAC: {total_cost_lac:.1f}\n"
    f"MPC: {total_cost_mpc:.1f}\n"
    f"LQR: {total_cost_lqr:.1f}\n"
)
ax2.text(
    0.99, 0.97, cost_text,
    transform=ax2.transAxes,
    fontsize=14,
    va='top', ha='right',
    bbox=dict(facecolor='white', alpha=0.92, edgecolor=color_lac, boxstyle='round,pad=0.4')
)

# 4. Learned Confidence (lambda)
lambda_lac_arr = np.array(lambda_lac[0]).flatten()
ax3 = fig.add_subplot(gs[3], sharex=ax0)
ax3.plot(np.arange(len(lambda_lac_arr)), lambda_lac_arr, label='LAC $\\lambda_t$', color=color_lac)
ax3.axvspan(attack_time, attack_end, color=color_attack_region, alpha=0.35, zorder=1)
ax3.axvline(x=attack_time, color=color_attack, linestyle='--', linewidth=2.0, zorder=2)
ax3.set_xlabel('Time Step $t$')
ax3.set_ylabel('Confidence $\\lambda_t$')
ax3.set_title('Learned Confidence Parameter')
ax3.set_ylim(-0.05, 1.05)
ax3.legend(fontsize=14, frameon=True, facecolor='white', edgecolor='gray')
ax3.grid(True, linestyle='--', alpha=0.8)

# Remove top/right spines for a cleaner look
for ax in [ax0, ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, T-1])

# Add subplot labels (A, B, C, D)
labels = ['A', 'B', 'C', 'D']
for i, ax in enumerate([ax0, ax1, ax2, ax3]):
    ax.text(-0.08, 1.08, labels[i], transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('robotic_arm_results.png', bbox_inches='tight')
plt.show()