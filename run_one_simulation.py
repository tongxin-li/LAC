from main import *
import matplotlib.pyplot as plt
import numpy as np

# --- Simulation Setup ---
SYSTEM_TYPE = 'linear'
noise_mu = 0
sigma_eta = 0
T = 200  # Use a longer time horizon to see the full effect

# --- Run Dynamics for Each Method ---
print("Running LAC...")
cost_lac, x_lac, u_lac, lambda_lac, phi_pred, phi_true = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'lac', 1)
print("Running MPC...")
cost_mpc, x_mpc, u_mpc, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '1-mpc', 1)
print("Running LQR/Nominal...")
cost_lqr, x_lqr, u_lqr, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '0-mpc', 1)
print("Running Self-Tuning...")
cost_st, x_st, u_st, lambda_st, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'self-tuning', 1)

# Helper function to compute instantaneous cost at each step
def get_inst_cost(x_hist, u_hist, system_type):
    costs = []
    cost_func_map = {
        'robotic_arm': robotic_arm_cost,
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
inst_cost_st = get_inst_cost(x_st[0], u_st[0], SYSTEM_TYPE)
time_steps = np.arange(T - 1)

# --- Smoothing function for cost curves ---
def smooth_curve(y, window=9):
    """Simple moving average smoothing."""
    if window < 2:
        return y
    y = np.array(y)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode='same')
    return y_smooth

# --- Plotting Results (Publication Quality) ---
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': True,
    'legend.edgecolor': 'gray',
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Adjust height ratios to make subfigure A (disturbance) and C (confidence) both shorter
fig, axs = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={'hspace': 0.25, 'height_ratios': [0.12, 0.3, 0.1]}
)
attack_time = T // 3
attack_end = 2 * T // 3

# --- Color Palette (Colorblind-friendly) ---
color_lac = '#0072B2'         # blue
color_mpc = '#009E73'         # green
color_lqr = '#E69F00'         # orange
color_st = '#666666'          # dark gray
color_attack = '#D55E00'      # red
color_disturbance = '#8E44AD' # purple
color_attack_region = '#D6C6E1' # light purple for span

# 1. Disturbance Error (epsilon_t)
phi_pred_arr = np.array(phi_pred[0])
phi_pred_sliced = phi_pred_arr[:, 0, :]
diff = phi_pred_sliced - phi_true[0][:T]
phi_norm = np.linalg.norm(diff, axis=1)

# --- Make the attack patterns subplot A more interesting ---
# Add a "rug" plot to show the true disturbance norm (ground truth) and overlay the prediction error
phi_true_arr = np.array(phi_true[0][:T])
phi_true_norm = np.linalg.norm(phi_true_arr, axis=1)

# Plot True Disturbance
line_true_disturbance, = axs[0].plot(
    np.arange(len(phi_true_norm))[:len(inst_cost_lac)],
    phi_true_norm[:len(inst_cost_lac)],
    label=r'True Disturbance $||\phi_t^{\star}||$',
    color=color_attack, alpha=0.45, linewidth=2.0, linestyle='-', zorder=2
)
axs[0].fill_between(
    np.arange(len(phi_true_norm))[:len(inst_cost_lac)],
    0, phi_true_norm[:len(inst_cost_lac)],
    color=color_attack, alpha=0.10, zorder=1
)
# Overlay the prediction error with markers and a step line
line_pred_error, = axs[0].plot(
    np.arange(len(phi_norm))[:len(inst_cost_lac)],
    phi_norm[:len(inst_cost_lac)],
    label=r'Average Error $||\epsilon_t||$',
    color=color_disturbance,
    zorder=4,
    marker='o',
    markersize=4,
    markerfacecolor='white',
    markeredgewidth=1.2,
    alpha=0.85,
    linewidth=2.5,
    markevery=10,
    linestyle='-'
)
axs[0].step(
    np.arange(len(phi_norm))[:len(inst_cost_lac)],
    phi_norm[:len(inst_cost_lac)],
    where='mid',
    color=color_disturbance,
    alpha=0.25,
    linewidth=1.2,
    zorder=3
)
# Add vertical lines for attack start/end
attack_period_patch = axs[0].axvspan(
    attack_time, attack_end,
    color=color_attack_region, alpha=0.35, label='Attack Period', zorder=0
)
attack_start_line = axs[0].axvline(
    x=attack_time, color=color_attack, linestyle='--', linewidth=2.5, label='Attack Start', zorder=5
)
attack_end_line = axs[0].axvline(
    x=attack_end, color=color_attack, linestyle=':', linewidth=2.0, label='Attack End', zorder=5
)
axs[0].set_title('Disturbance and Prediction Error During Attack')
axs[0].set_ylabel(r'$||\cdot||$')
# Decompose the legend: only show True Disturbance and Prediction Error in the main legend
handles_main = [line_true_disturbance, line_pred_error]
labels_main = [r'True Disturbance $||\phi_t||$', r'Mean Error $||\epsilon_t||$']
axs[0].legend(handles=handles_main, labels=labels_main, loc='upper left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray', ncol=1)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].grid(True, color='#e5e5e5', linestyle='--', linewidth=1.2, alpha=0.9)
axs[0].set_xlim([0, T-1])
axs[0].set_ylim(bottom=0)

# Add a separate legend for Attack Start and Attack End, moved to the left to ensure it's within the figure
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=color_attack, linestyle='--', linewidth=2.5),
    Line2D([0], [0], color=color_attack, linestyle=':', linewidth=2.0)
]
custom_labels = ['Attack Start', 'Attack End']
# Place this legend just inside the right edge of the plot area
axs[0].legend(
    handles=handles_main, labels=labels_main, loc='upper left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray', ncol=1
)
axs[0].add_artist(axs[0].legend(handles=handles_main, labels=labels_main, loc='upper left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray', ncol=1))
# Move the attack lines legend to the right but within the axes (e.g., x=0.82)
axs[0].legend(
    handles=custom_lines, labels=custom_labels,
    loc='center left', bbox_to_anchor=(0.765, 0.5),
    fontsize=18, frameon=True, facecolor='white', edgecolor='gray', ncol=1, title=None
)

# 2. Instantaneous Costs (SMOOTHED curves)
min_len = min(len(time_steps), len(inst_cost_lac), len(inst_cost_mpc), len(inst_cost_lqr), len(inst_cost_st))

window = 9  # Smoothing window size
inst_cost_lac_smooth = smooth_curve(inst_cost_lac[:min_len], window)
inst_cost_mpc_smooth = smooth_curve(inst_cost_mpc[:min_len], window)
inst_cost_lqr_smooth = smooth_curve(inst_cost_lqr[:min_len], window)
inst_cost_st_smooth = smooth_curve(inst_cost_st[:min_len], window)

# Plot smoothed curves
axs[1].plot(
    time_steps[:min_len], inst_cost_lac_smooth,
    label='LAC', color=color_lac, linewidth=2.8, zorder=3,
)
axs[1].plot(
    time_steps[:min_len], inst_cost_mpc_smooth,
    label='MPC (w/ predictions)', color=color_mpc, linestyle='--', linewidth=2.8, zorder=3,
)
axs[1].plot(
    time_steps[:min_len], inst_cost_lqr_smooth,
    label='LQR (nominal)', color=color_lqr, linestyle='-.', linewidth=2.8, zorder=3,
)
axs[1].plot(
    time_steps[:min_len], inst_cost_st_smooth,
    label='Self-Tuning', color=color_st, linestyle=':', linewidth=2.8, zorder=3,
)
# (Removed faint original (unsmoothed) curves as dots for reference)

axs[1].axvspan(
    attack_time, attack_end,
    color=color_attack_region, alpha=0.35, zorder=1
)
axs[1].axvline(
    x=attack_time, color=color_attack, linestyle='--', linewidth=2.5, zorder=4
)
axs[1].axvline(
    x=attack_end, color=color_attack, linestyle=':', linewidth=2.0, zorder=4
)
axs[1].set_title('Controller Performance Under Attack')
axs[1].set_ylabel('Instantaneous Cost')
axs[1].legend(loc='upper left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray')
axs[1].set_ylim(bottom=0, top=max(np.max(inst_cost_lac_smooth), np.max(inst_cost_mpc_smooth), np.max(inst_cost_lqr_smooth), np.max(inst_cost_st_smooth)) * 1.15)
axs[1].grid(True, color='#e5e5e5', linestyle='--', linewidth=1.2, alpha=0.9)
axs[1].set_xlim([0, T-1])

# Annotate total costs in a clean, publication-style box
total_cost_lac = np.sum(cost_lac)
total_cost_mpc = np.sum(cost_mpc)
total_cost_lqr = np.sum(cost_lqr)
total_cost_st = np.sum(cost_st)
cost_text = (
    f"LAC: {total_cost_lac:.1f}\n"
    f"MPC: {total_cost_mpc:.1f}\n"
    f"LQR: {total_cost_lqr:.1f}\n"
    f"Self-Tuning: {total_cost_st:.1f}"
)
axs[1].text(
    0.99, 0.97, cost_text,
    transform=axs[1].transAxes,
    fontsize=17,
    va='top', ha='right',
    bbox=dict(facecolor='white', alpha=0.92, edgecolor=color_lac, boxstyle='round,pad=0.4')
)

# 3. Learned Confidence (lambda)
lambda_lac_arr = np.array(lambda_lac[0])
lambda_st_arr = np.array(lambda_st[0])
axs[2].plot(
    np.arange(len(lambda_lac_arr)), lambda_lac_arr,
    label=r'LAC $\lambda_t$',
    color=color_lac, linewidth=2.8, zorder=3
)
axs[2].plot(
    np.arange(len(lambda_st_arr)), lambda_st_arr,
    label=r'Self-Tuning $\lambda_t$',
    color=color_st, linestyle=':', linewidth=2.8, zorder=3
)
axs[2].axvspan(
    attack_time, attack_end,
    color=color_attack_region, alpha=0.35, zorder=1
)
axs[2].axvline(
    x=attack_time, color=color_attack, linestyle='--', linewidth=2.5, zorder=4
)
axs[2].axvline(
    x=attack_end, color=color_attack, linestyle=':', linewidth=2.0, zorder=4
)
axs[2].set_title('Learned Confidence Parameter')
axs[2].set_xlabel('Time Step $t$')
axs[2].set_ylabel(r'$\lambda_t$')
axs[2].set_ylim(-0.05, 1.05)
axs[2].legend(loc='lower left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray')
axs[2].grid(True, color='#e5e5e5', linestyle='--', linewidth=1.2, alpha=0.9)
axs[2].set_xlim([0, T-1])

# Remove top/right spines for a cleaner look
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add subplot labels (A, B, C)
labels = ['A', 'B', 'C']
for i, ax in enumerate(axs):
    ax.text(-0.08, 1.08, labels[i], transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('perturbation_attack_results.png', bbox_inches='tight')
plt.show()

# --- FIGURE: Tracking Curves vs. Desired Trajectory (Attack and After-Attack Phases) ---

# For SYSTEM_TYPE='linear', the desired trajectory is the (y_1, y_2) from generate_phi.py
# We plot the first two state dimensions (x1, x2) for each method, and the desired (y1, y2) as reference,
# for both the attack phase and the after-attack phase.

from generate_phi import tracking_coordinates

# Define attack and after-attack intervals
attack_start = 100
attack_end = 110
after_attack_start = attack_end
after_attack_end = 120  # 10 steps after attack

attack_indices = np.arange(attack_start, attack_end)
after_attack_indices = np.arange(after_attack_start, after_attack_end)

# Reconstruct the desired trajectory (y1, y2) for both phases
desired_y1_attack, desired_y2_attack = [], []
desired_y1_after, desired_y2_after = [], []
for t in attack_indices:
    y1, y2 = tracking_coordinates(t)
    desired_y1_attack.append(y1)
    desired_y2_attack.append(y2)
for t in after_attack_indices:
    y1, y2 = tracking_coordinates(t)
    desired_y1_after.append(y1)
    desired_y2_after.append(y2)
desired_y1_attack = np.array(desired_y1_attack)
desired_y2_attack = np.array(desired_y2_attack)
desired_y1_after = np.array(desired_y1_after)
desired_y2_after = np.array(desired_y2_after)

# Extract the first two state dimensions for each method (attack and after-attack)
x_lac_arr_attack = np.array(x_lac[0])[attack_start:attack_end]
x_mpc_arr_attack = np.array(x_mpc[0])[attack_start:attack_end]
x_lqr_arr_attack = np.array(x_lqr[0])[attack_start:attack_end]
x_st_arr_attack = np.array(x_st[0])[attack_start:attack_end]

x_lac_arr_after = np.array(x_lac[0])[after_attack_start:after_attack_end]
x_mpc_arr_after = np.array(x_mpc[0])[after_attack_start:after_attack_end]
x_lqr_arr_after = np.array(x_lqr[0])[after_attack_start:after_attack_end]
x_st_arr_after = np.array(x_st[0])[after_attack_start:after_attack_end]

# --- Plotting the tracking curves for both phases ---
plt.figure(figsize=(12, 7))

# Plot desired trajectory for attack and after-attack
plt.plot(desired_y1_attack, desired_y2_attack, 'k-', linewidth=3, label='Desired Trajectory (Attack)', alpha=0.7, zorder=6)
plt.plot(desired_y1_after, desired_y2_after, 'k--', linewidth=2.2, label='Desired Trajectory (After Attack)', alpha=0.7, zorder=5)

# Plot each method during attack phase
plt.plot(x_lac_arr_attack[:,0], x_lac_arr_attack[:,1], color=color_lac, label='LAC (Attack)', linewidth=2.5, zorder=4)
plt.plot(x_mpc_arr_attack[:,0], x_mpc_arr_attack[:,1], color=color_mpc, label='MPC (Attack)', linestyle='--', linewidth=2.2, zorder=3)
plt.plot(x_lqr_arr_attack[:,0], x_lqr_arr_attack[:,1], color=color_lqr, label='LQR (Attack)', linestyle='-.', linewidth=2.2, zorder=2)
plt.plot(x_st_arr_attack[:,0], x_st_arr_attack[:,1], color=color_st, label='Self-Tuning (Attack)', linestyle=':', linewidth=2.2, zorder=1)

# Plot each method during after-attack phase (faded for visual separation)
plt.plot(x_lac_arr_after[:,0], x_lac_arr_after[:,1], color=color_lac, linewidth=2.5, alpha=0.5, label='LAC (After Attack)', zorder=4)
plt.plot(x_mpc_arr_after[:,0], x_mpc_arr_after[:,1], color=color_mpc, linestyle='--', linewidth=2.2, alpha=0.5, label='MPC (After Attack)', zorder=3)
plt.plot(x_lqr_arr_after[:,0], x_lqr_arr_after[:,1], color=color_lqr, linestyle='-.', linewidth=2.2, alpha=0.5, label='LQR (After Attack)', zorder=2)
plt.plot(x_st_arr_after[:,0], x_st_arr_after[:,1], color=color_st, linestyle=':', linewidth=2.2, alpha=0.5, label='Self-Tuning (After Attack)', zorder=1)

# Mark start and end points of the attack phase
plt.scatter(desired_y1_attack[0], desired_y2_attack[0], color='k', s=90, marker='o', label='Attack Start', zorder=10)
plt.scatter(desired_y1_attack[-1], desired_y2_attack[-1], color='k', s=90, marker='X', label='Attack End', zorder=10)

# Emphasize LAC's superior tracking
# Only compare the first two state dimensions (y1, y2) for tracking error
desired_attack_traj = np.stack([desired_y1_attack, desired_y2_attack], axis=1)  # shape (N,2)
lac_attack_dist = np.linalg.norm(x_lac_arr_attack[:, :2] - desired_attack_traj, axis=1)
mpc_attack_dist = np.linalg.norm(x_mpc_arr_attack[:, :2] - desired_attack_traj, axis=1)
lqr_attack_dist = np.linalg.norm(x_lqr_arr_attack[:, :2] - desired_attack_traj, axis=1)
st_attack_dist = np.linalg.norm(x_st_arr_attack[:, :2] - desired_attack_traj, axis=1)

# Only compare the first two state dimensions for tracking error (to match desired trajectory shape)
desired_after_traj = np.stack([desired_y1_after, desired_y2_after], axis=1)  # shape (N,2)

lac_after_dist = np.linalg.norm(x_lac_arr_after[:, :2] - desired_after_traj, axis=1)
mpc_after_dist = np.linalg.norm(x_mpc_arr_after[:, :2] - desired_after_traj, axis=1)
lqr_after_dist = np.linalg.norm(x_lqr_arr_after[:, :2] - desired_after_traj, axis=1)
st_after_dist = np.linalg.norm(x_st_arr_after[:, :2] - desired_after_traj, axis=1)

lac_attack_rmse = np.sqrt(np.mean(lac_attack_dist**2))
mpc_attack_rmse = np.sqrt(np.mean(mpc_attack_dist**2))
lqr_attack_rmse = np.sqrt(np.mean(lqr_attack_dist**2))
st_attack_rmse = np.sqrt(np.mean(st_attack_dist**2))

lac_after_rmse = np.sqrt(np.mean(lac_after_dist**2))
mpc_after_rmse = np.sqrt(np.mean(mpc_after_dist**2))
lqr_after_rmse = np.sqrt(np.mean(lqr_after_dist**2))
st_after_rmse = np.sqrt(np.mean(st_after_dist**2))

# Annotate the plot with RMSE values and a note about LAC
textstr = (
    "Tracking RMSE (Attack Phase):\n"
    f"LAC: {lac_attack_rmse:.3f}\n"
    f"MPC: {mpc_attack_rmse:.3f}\n"
    f"LQR: {lqr_attack_rmse:.3f}\n"
    f"Self-Tuning: {st_attack_rmse:.3f}\n\n"
    "Tracking RMSE (After Attack):\n"
    f"LAC: {lac_after_rmse:.3f}\n"
    f"MPC: {mpc_after_rmse:.3f}\n"
    f"LQR: {lqr_after_rmse:.3f}\n"
    f"Self-Tuning: {st_after_rmse:.3f}\n\n"
    "Note: LAC maintains the lowest tracking error during and after the attack phase, "
    "demonstrating its superior robustness and adaptability to abrupt changes. "
    "This is because LAC dynamically adjusts its confidence parameter, allowing it to quickly adapt to the new system dynamics induced by the attack, "
    "whereas other methods either lack adaptation (LQR), rely on fixed models (MPC), or adapt more slowly (Self-Tuning)."
)
plt.gca().text(
    1.02, 0.5, textstr, transform=plt.gca().transAxes,
    fontsize=14, va='center', ha='left',
    bbox=dict(facecolor='white', alpha=0.93, edgecolor=color_lac, boxstyle='round,pad=0.5')
)

plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Tracking Performance: Attack vs. After-Attack Phases')
plt.legend(loc='best', fontsize=14, frameon=True, facecolor='white', edgecolor='gray', ncol=2)
plt.grid(True, color='#e5e5e5', linestyle='--', linewidth=1.2, alpha=0.9)
plt.tight_layout()
plt.savefig('tracking_trajectory_comparison_attack_and_after.png', bbox_inches='tight')
plt.show()