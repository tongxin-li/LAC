from main import *
import matplotlib.pyplot as plt
import numpy as np

# --- Simulation Setup ---
SYSTEM_TYPE = 'robotic_arm'
noise_mu = 0
sigma_eta = 0
T = 200  # Time horizon

# --- Run Dynamics for Each Method ---
print("Running Self-Tuning...")
cost_st, x_st, u_st, lambda_st, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'self-tuning', 1)
print("Running LAC...")
cost_lac, x_lac, u_lac, lambda_lac, phi_pred, phi_true = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, 'lac', 1)
print("Running P-MPC...")
cost_mpc, x_mpc, u_mpc, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '1-mpc', 1)
print("Running N-MPC...")
cost_lqr, x_lqr, u_lqr, _, _, _ = run_dynamics(noise_mu, sigma_eta, SYSTEM_TYPE, '0-mpc', 1)

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

# --- Plotting Style: revert to default font, keep seaborn darkgrid ---
import matplotlib as mpl
import seaborn as sns

# Use seaborn darkgrid, but revert to default font
sns.set_theme(context='paper', style='darkgrid', font_scale=1.18, rc={
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.8,
    'lines.linewidth': 2.5,
    'axes.edgecolor': '#181818',
    'axes.labelcolor': '#181818',
    'xtick.color': '#181818',
    'ytick.color': '#181818',
    'legend.frameon': True,
    'legend.framealpha': 0.98,
    'legend.fancybox': True,
    'legend.edgecolor': '#181818',
    'savefig.dpi': 400,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Remove font.family and font.serif to revert to default
})

attack_time = T // 3
attack_end = 2 * T // 3

# --- Color Palette (Colorblind-friendly, with extra pop) ---
color_lac = sns.color_palette("colorblind")[0]
color_mpc = sns.color_palette("colorblind")[1]
color_lqr = sns.color_palette("colorblind")[2]
color_st = sns.color_palette("colorblind")[4]
color_attack = sns.color_palette("colorblind")[5]
color_attack_region = "#e5e5e5"
highlight_color = "#FFD700"  # gold for highlight

# Helper for moving average smoothing
def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

# Prepare data
x_lac_arr = np.array(x_lac[0]).flatten()
x_mpc_arr = np.array(x_mpc[0]).flatten()
x_lqr_arr = np.array(x_lqr[0]).flatten()
x_st_arr = np.array(x_st[0]).flatten()

u_lac_arr = np.array(u_lac[0]).flatten()
u_mpc_arr = np.array(u_mpc[0]).flatten()
u_lqr_arr = np.array(u_lqr[0]).flatten()
u_st_arr = np.array(u_st[0]).flatten()

window = 7
inst_cost_lac_smooth = moving_average(inst_cost_lac, window)
inst_cost_mpc_smooth = moving_average(inst_cost_mpc, window)
inst_cost_lqr_smooth = moving_average(inst_cost_lqr, window)
inst_cost_st_smooth = moving_average(inst_cost_st, window)

# --- Custom Figure Layout: 2x2, with C (cost) at lower left, D (confidence) at lower right (smaller) ---
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(9.2, 5.1))
# Make lower right (D) smaller by adjusting width_ratios and height_ratios
gs = GridSpec(2, 2, height_ratios=[1, 0.8], width_ratios=[1, 0.55], hspace=0.32, wspace=0.22)

# Helper to draw a colored frame for the attack region
def draw_attack_frame(ax, attack_time, attack_end, color, linewidth=2, linestyle='--', alpha=0.85, zorder=10, label=None):
    ylim = ax.get_ylim()
    rect = Rectangle(
        (attack_time, ylim[0]),
        attack_end - attack_time,
        ylim[1] - ylim[0],
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        label=label
    )
    ax.add_patch(rect)

# 1. State (x) over time (A)
ax0 = fig.add_subplot(gs[0, 0])
l0, = ax0.plot(np.arange(len(x_lac_arr)), x_lac_arr, label='LAC', color=color_lac, linestyle='-', marker='o', markevery=20, markersize=4.5, zorder=3)
l1, = ax0.plot(np.arange(len(x_mpc_arr)), x_mpc_arr, label='P-MPC', color=color_mpc, linestyle='--', marker='s', markevery=20, markersize=4.5, zorder=3)
l2, = ax0.plot(np.arange(len(x_lqr_arr)), x_lqr_arr, label='N-MPC', color=color_lqr, linestyle='-.', marker='^', markevery=20, markersize=4.5, zorder=3)
l3, = ax0.plot(np.arange(len(x_st_arr)), x_st_arr, label='Self-Tuning', color=color_st, linestyle=':', marker='D', markevery=20, markersize=4.5, zorder=3)
# Draw colored frame for attack region
draw_attack_frame(ax0, attack_time, attack_end, color_attack, linewidth=2, linestyle='--', alpha=0.85, zorder=10, label='Attack Region')
ax0.axvline(x=attack_time, color=color_attack, linestyle=':', linewidth=2.5, zorder=1)
ax0.set_ylabel('State $x$', labelpad=2)
ax0.set_title('Robotic Arm State', pad=4, fontweight='bold')
ax0.set_xlim([0, T-1])
ax0.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

# 2. Control Input (u) over time (B)
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(np.arange(len(u_lac_arr)), u_lac_arr, color=color_lac, linestyle='-', marker='o', markevery=20, markersize=4.5, zorder=3)
ax1.plot(np.arange(len(u_mpc_arr)), u_mpc_arr, color=color_mpc, linestyle='--', marker='s', markevery=20, markersize=4.5, zorder=3)
ax1.plot(np.arange(len(u_lqr_arr)), u_lqr_arr, color=color_lqr, linestyle='-.', marker='^', markevery=20, markersize=4.5, zorder=3)
ax1.plot(np.arange(len(u_st_arr)), u_st_arr, color=color_st, linestyle=':', marker='D', markevery=20, markersize=4.5, zorder=3)
# Draw colored frame for attack region
draw_attack_frame(ax1, attack_time, attack_end, color_attack, linewidth=2, linestyle='--', alpha=0.85, zorder=10)
ax1.axvline(x=attack_time, color=color_attack, linestyle=':', linewidth=2.5, zorder=1)
ax1.set_ylabel('Control $u$', labelpad=2)
ax1.set_title('Control Input', pad=4, fontweight='bold')
ax1.set_xlim([0, T-1])
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

# 3. Lower left: Cost (C)
ax2 = fig.add_subplot(gs[1, 0])
min_len = min(len(time_steps), len(inst_cost_lac_smooth), len(inst_cost_mpc_smooth), len(inst_cost_lqr_smooth), len(inst_cost_st_smooth))
cost_lac_line, = ax2.plot(time_steps[:min_len], inst_cost_lac_smooth[:min_len], color=color_lac, linestyle='-', marker='o', markevery=20, markersize=4, label='LAC', zorder=3)
cost_mpc_line, = ax2.plot(time_steps[:min_len], inst_cost_mpc_smooth[:min_len], color=color_mpc, linestyle='--', marker='s', markevery=20, markersize=4, label='MPC', zorder=3)
cost_lqr_line, = ax2.plot(time_steps[:min_len], inst_cost_lqr_smooth[:min_len], color=color_lqr, linestyle='-.', marker='^', markevery=20, markersize=4, label='LQR', zorder=3)
cost_st_line, = ax2.plot(time_steps[:min_len], inst_cost_st_smooth[:min_len], color=color_st, linestyle=':', marker='D', markevery=20, markersize=4, label='Self-Tuning', zorder=3)
# Draw colored frame for attack region
draw_attack_frame(ax2, attack_time, attack_end, color_attack, linewidth=2, linestyle='--', alpha=0.85, zorder=10)
ax2.axvline(x=attack_time, color=color_attack, linestyle=':', linewidth=2.5, zorder=1)
ax2.set_ylabel('Inst. Cost', labelpad=2)
ax2.set_xlabel('Time $t$', labelpad=2)
ax2.set_xlim([0, T-1])
ax2.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
# Add title for subfigure C (Cost) on top
ax2.set_title('Cost', pad=4, fontweight='bold')
# Annotate total costs (inset box, upper right, with gold border)
total_cost_lac = np.sum(cost_lac)
total_cost_mpc = np.sum(cost_mpc)
total_cost_lqr = np.sum(cost_lqr)
total_cost_st = np.sum(cost_st)
cost_text = (
    f"LAC: {total_cost_lac:.1f}\n"
    f"P-MPC: {total_cost_mpc:.1f}\n"
    f"N-MPC: {total_cost_lqr:.1f}\n"
    f"Self-Tuning: {total_cost_st:.1f}"
)
ax2.text(
    0.99, 0.9, cost_text,
    transform=ax2.transAxes,
    fontsize=10,
    va='top', ha='right',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor=highlight_color, boxstyle='round,pad=0.28', linewidth=1.7)
)

# 4. Lower right: Confidence (lambda) (D), smaller
ax3 = fig.add_subplot(gs[1, 1])
lambda_lac_arr = np.array(lambda_lac[0]).flatten()
lambda_st_arr = np.array(lambda_st[0]).flatten()
lambda_lac_line, = ax3.plot(np.arange(len(lambda_lac_arr)), lambda_lac_arr, color=color_lac, linestyle='-', linewidth=2.1, alpha=0.85, label='LAC $\lambda_t$', zorder=4)
lambda_st_line, = ax3.plot(np.arange(len(lambda_st_arr)), lambda_st_arr, color=color_st, linestyle=':', linewidth=2.1, alpha=0.85, label='Self-Tuning $\lambda_t$', zorder=4)
# Draw colored frame for attack region
draw_attack_frame(ax3, attack_time, attack_end, color_attack, linewidth=2, linestyle='--', alpha=0.85, zorder=10)
ax3.axvline(x=attack_time, color=color_attack, linestyle=':', linewidth=2.5, zorder=1)
ax3.set_ylabel('Confidence $\lambda_t$', labelpad=2)
ax3.set_xlabel('Time $t$', labelpad=2)
ax3.set_ylim(-0.05, 1.05)
ax3.set_xlim([0, T-1])
ax3.tick_params(axis='y', labelsize=10)
# Add title for subfigure D (Confidence) on top
ax3.set_title('Confidence', pad=4, fontweight='bold')

# Remove top/right spines for a clean academic look
for ax in [ax0, ax1, ax2, ax3]:
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlim([0, T-1])

# Add subplot labels (A, B, C, D) in bold, academic style
labels = ['A', 'B', 'C', 'D']
axes_list = [ax0, ax1, ax2, ax3]
for i, ax in enumerate(axes_list):
    ax.text(-0.13, 1.13, labels[i], transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
# Remove old in-axes "Cost" and "Confidence" text for C and D
# (No longer needed since we use set_title above)

# Remove legends from all subplots
for ax in [ax0, ax1, ax2, ax3]:
    if hasattr(ax, "legend_") and ax.legend_:
        ax.legend_.remove()

# Create a single shared legend below the main plot
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color=color_lac, linestyle='-', marker='o', label='LAC', markersize=6, linewidth=2.5),
    Line2D([], [], color=color_mpc, linestyle='--', marker='s', label='P-MPC', markersize=6, linewidth=2.5),
    Line2D([], [], color=color_lqr, linestyle='-.', marker='^', label='N-MPC', markersize=6, linewidth=2.5),
    Line2D([], [], color=color_st, linestyle=':', marker='D', label='Self-Tuning', markersize=6, linewidth=2.5),
    Line2D([0], [0], color=color_attack, alpha=1, linewidth=3.5, linestyle='--', label='Attack Region'),
    Line2D([], [], color=color_lac, linestyle='-', linewidth=2.5, label='LAC $\lambda_t$'),
    Line2D([], [], color=color_st, linestyle=':', linewidth=2.5, label='Self-Tuning $\lambda_t$')
]
legend_labels = [
    'LAC', 'P-MPC', 'N-MPC', 'Self-Tuning', 'Attack Region', 'LAC $\lambda_t$', 'Self-Tuning $\lambda_t$'
]

fig.legend(handles, legend_labels, loc='lower center', ncol=4, frameon=True, fancybox=True, edgecolor='#181818', fontsize=11, bbox_to_anchor=(0.5, -0.04), handletextpad=0.8, columnspacing=1.3, borderaxespad=0.7)

plt.tight_layout(pad=0.8, rect=[0, 0.08, 1, 1])
plt.subplots_adjust(bottom=0.17)
plt.savefig('robotic_arm_results_2col_fancy.pdf', bbox_inches='tight')
plt.show()

# # --- NEW FIGURE: Visualize the robot arms for the compared methods ---

# import matplotlib.patches as mpatches

# def plot_robotic_arm(ax, theta, color, label=None, base=(0,0), arm_length=1.0, zorder=2, lw=3):
#     """
#     Draw a simple 1-link robotic arm (scalar state = angle theta).
#     """
#     x0, y0 = base
#     x1 = x0 + arm_length * np.cos(theta)
#     y1 = y0 + arm_length * np.sin(theta)
#     ax.plot([x0, x1], [y0, y1], color=color, lw=lw, marker='o', markersize=8, zorder=zorder, label=label)
#     # Draw a circle at the end effector
#     ax.plot(x1, y1, 'o', color=color, markersize=12, zorder=zorder+1)
#     return (x1, y1)

# # Choose time points to visualize (start, before attack, during attack, after attack, end)
# time_points = [
#     0,
#     attack_time-5,
#     attack_time+5,
#     attack_end-5,
#     T-2
# ]
# time_labels = [
#     "Start",
#     "Pre-Attack",
#     "Early Attack",
#     "Late Attack",
#     "End"
# ]

# # Prepare thetas for each method at each time point
# thetas = {
#     'LAC': [x_lac_arr[t] for t in time_points],
#     'MPC': [x_mpc_arr[t] for t in time_points],
#     'LQR': [x_lqr_arr[t] for t in time_points],
#     'Self-Tuning': [x_st_arr[t] for t in time_points]
# }
# arm_colors = {
#     'LAC': color_lac,
#     'MPC': color_mpc,
#     'LQR': color_lqr,
#     'Self-Tuning': color_st
# }

# # Create the figure
# fig2, axes2 = plt.subplots(1, len(time_points), figsize=(2.1*len(time_points), 3.2), sharey=True)
# if len(time_points) == 1:
#     axes2 = [axes2]

# for i, (ax, t, tlabel) in enumerate(zip(axes2, time_points, time_labels)):
#     # Draw a faint circle for workspace
#     circle = mpatches.Circle((0,0), 1.0, color='gray', alpha=0.13, zorder=0)
#     ax.add_patch(circle)
#     # Draw each method's arm
#     for method, color in arm_colors.items():
#         plot_robotic_arm(ax, thetas[method][i], color, label=method, lw=3)
#     ax.set_xlim(-1.15, 1.15)
#     ax.set_ylim(-1.15, 1.15)
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f"{tlabel}\n$t={t}$", fontsize=13, fontweight='bold', pad=7)
#     # Draw a small dot at the base
#     ax.plot(0, 0, 'ko', markersize=6, zorder=5)
#     # Remove spines
#     for spine in ax.spines.values():
#         spine.set_visible(False)

# # Custom legend
# custom_lines = [
#     Line2D([0], [0], color=color_lac, lw=3, marker='o', markersize=8, label='LAC'),
#     Line2D([0], [0], color=color_mpc, lw=3, marker='o', markersize=8, label='MPC'),
#     Line2D([0], [0], color=color_lqr, lw=3, marker='o', markersize=8, label='LQR'),
#     Line2D([0], [0], color=color_st, lw=3, marker='o', markersize=8, label='Self-Tuning')
# ]
# fig2.legend(custom_lines, ['LAC', 'MPC', 'LQR', 'Self-Tuning'],
#             loc='lower center', ncol=4, frameon=True, fancybox=True, edgecolor='#181818',
#             fontsize=12, bbox_to_anchor=(0.5, 0), handletextpad=0.8, columnspacing=1.3, borderaxespad=0.7)

# fig2.suptitle("Robotic Arm Visualization at Key Time Points", fontsize=15, fontweight='bold', y=1.05)
# plt.tight_layout(pad=1.0, rect=[0, 0.08, 1, 1])
# plt.subplots_adjust(bottom=0.18, top=0.85)
# plt.savefig('robotic_arm_visualization_methods.pdf', bbox_inches='tight')
# plt.show()