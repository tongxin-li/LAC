import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_robotic_arm_illustration():
    """
    Generates and displays an illustration of the single-joint robotic arm
    with its dynamic components labeled.
    """
    # --- Setup the plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off') # Hide the axes for a cleaner look
    ax.set_title("Illustration of Single-Joint Robotic Arm Dynamics", fontsize=16, pad=20)

    # --- Define Arm Parameters ---
    pivot = (0, 0)
    arm_length = 1.0
    # Set an example angle for the illustration (45 degrees)
    angle_rad = np.deg2rad(45) 
    
    # Calculate arm endpoint coordinates
    end_point = (arm_length * np.sin(angle_rad), -arm_length * np.cos(angle_rad))

    # --- Draw the Arm Components ---
    # Pivot point (motor)
    ax.plot(pivot[0], pivot[1], 'ko', markersize=15, zorder=5)
    ax.text(pivot[0], pivot[1] + 0.1, 'Pivot (Motor)', ha='center', fontsize=12)

    # Arm link
    ax.plot([pivot[0], end_point[0]], [pivot[1], end_point[1]], 'k-', linewidth=5, zorder=4)
    ax.text(end_point[0] / 2, end_point[1] / 2 - 0.1, 'Arm', ha='center', fontsize=12)

    # Mass at the end
    ax.plot(end_point[0], end_point[1], 'o', color='steelblue', markersize=25, zorder=5)
    ax.text(end_point[0], end_point[1], 'Mass', color='white', ha='center', va='center', fontsize=10)

    # --- Illustrate Dynamic Forces and Torques ---
    # State (Angle x_t)
    angle_arc = patches.Arc(pivot, 0.4, 0.4, angle=270, theta1=0, theta2=np.rad2deg(angle_rad),
                            color='black', linestyle='--', linewidth=1.5)
    ax.add_patch(angle_arc)
    ax.text(0.25, -0.2, r'$x_t$', fontsize=16, color='black')

    # Control Torque (u_t)
    control_arc = patches.FancyArrowPatch(posA=(0.3, 0.1), posB=(0.1, 0.3),
                                          connectionstyle="arc3,rad=-0.5",
                                          color="green", arrowstyle='->,head_length=10,head_width=6',
                                          linewidth=2.5)
    ax.add_patch(control_arc)
    ax.text(0.35, 0.35, r'Control Torque ($u_t$)', fontsize=14, color='green', ha='center')
    ax.text(0.35, 0.25, r'$0.2 u_t e^{-|x_t|}$', fontsize=12, color='green', ha='center')


    # Gravity (g)
    ax.arrow(end_point[0], end_point[1], 0, -0.4, head_width=0.05, head_length=0.1, fc='darkred', ec='darkred', linewidth=2)
    ax.text(end_point[0] + 0.05, end_point[1] - 0.5, 'Gravity (g)', fontsize=14, color='darkred', va='top')
    ax.text(end_point[0] + 0.05, end_point[1] - 0.6, r'$0.5 \sin(x_t)$', fontsize=12, color='darkred', va='top')


    # Disturbance (phi_t)
    ax.arrow(end_point[0] - 0.5, end_point[1], 0.3, 0, head_width=0.05, head_length=0.1,
             fc='purple', ec='purple', linewidth=2, linestyle='--')
    ax.text(end_point[0] - 0.5, end_point[1] + 0.1, r'Disturbance ($\phi_t$)', fontsize=14, color='purple', ha='left')

    plt.show()

if __name__ == '__main__':
    plot_robotic_arm_illustration()