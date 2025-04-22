import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_t = np.pi / 6  # Angle in radians (30 degrees)
arm_length = 1.0  # Length of the arm
base_position = (0, 0)  # Base of the arm

# Compute end position of the arm
end_x = arm_length * np.sin(x_t)
end_y = -arm_length * np.cos(x_t)  # Negative because y-axis points up

# Plotting
plt.figure(figsize=(6, 6))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', adjustable='box')

# Plot base
plt.plot(base_position[0], base_position[1], 'ko', markersize=10, label='Base')

# Plot reference line (vertical)
plt.plot([0, 0], [-arm_length, 0], 'r--', label='Reference (Vertical)')

# Plot arm
plt.plot([base_position[0], end_x], [base_position[1], end_y], 'b-', linewidth=3, label='Arm')

# Plot angle arc
theta = np.linspace(0, x_t, 100)
arc_x = 0.3 * np.sin(theta)
arc_y = -0.3 * np.cos(theta)
plt.plot(arc_x, arc_y, 'k-', label='Angle $x_t$')

# Annotate angle
plt.text(0.15, -0.15, '$x_t$', fontsize=12)

# Plot control torque arrow
torque_direction = (np.cos(x_t), np.sin(x_t))  # Tangential direction
plt.arrow(end_x * 0.5, end_y * 0.5, 0.2 * torque_direction[0], 0.2 * torque_direction[1],
          head_width=0.1, head_length=0.1, fc='g', ec='g', label='Control Torque $u_t$')

# Plot disturbance arrow
disturbance_direction = (-np.sin(x_t), -np.cos(x_t))  # Radial direction
plt.arrow(end_x, end_y, 0.2 * disturbance_direction[0], 0.2 * disturbance_direction[1],
          head_width=0.1, head_length=0.1, fc='orange', ec='orange', label='Disturbance $\phi_t$')

# Labels and title
plt.title('Robotic Arm Angle Dynamics: System Figure', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True)
plt.show()