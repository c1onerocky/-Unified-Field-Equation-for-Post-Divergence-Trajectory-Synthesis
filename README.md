import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Setup Time and Parameters
t = np.linspace(0, 10, 100)
td = 5  
alpha = [0.5, 0.3, 0.2] 

# 2. Pre-Divergence (Position + Rotation)
x0, y0, z0 = 0.5 * t**2, 0.2 * t, t * 10
# Initial orientation: mostly stable
roll0, pitch0, yaw0 = 0.1 * t, 0.05 * t, 0.02 * t

# 3. Post-Divergence Complex Synthesis
def generate_complex_physics(t, td, x0, y0, z0):
    mask = t >= td
    post_td = t[mask] - td
    branches = []
    
    for i in range(len(alpha)):
        # Translational Branches (Real + Imaginary i)
        xi = x0[mask] + (i * post_td**1.5)
        yi = y0[mask] + (np.sin(i * post_td) * 2)
        zi = z0[mask] + (1 * post_td)
        im_pos = np.exp(0.4 * post_td) * (i + 1) # Position Tendency
        
        # Rotational Branches (Theta_i_complex)
        # Real = Actual tilt, Imaginary = The "Wobble" phase
        roll_i = roll0[mask] + (0.2 * i * post_td)
        im_roll = np.sin(post_td * (i+1)) * post_td # The complex "wobble"
        
        branches.append({
            'pos': (xi, yi, zi, im_pos),
            'rot': (roll_i, im_roll)
        })
    return branches

branches = generate_complex_physics(t, td, x0, y0, z0)

# 4. Visualization
fig = plt.figure(figsize=(14, 7))



# Subplot 1: 3D Trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x0[t<td], y0[t<td], z0[t<td], 'k--', label="Stable Entry")
colors = ['magento', 'cyan', 'orange']

for i, b in enumerate(branches):
    xi, yi, zi, im = b['pos']
    ax1.plot(xi, yi, zi, color=colors[i%3], label=f"Branch {i+1}")
    ax1.scatter(xi, yi, zi, s=im*5, color=colors[i%3], alpha=0.05) # Uncertainty aura

ax1.set_title("Complex Trajectory (Real + Imaginary)")
ax1.legend()

# Subplot 2: Rotational Instability (The "Wobble")
ax2 = fig.add_subplot(122)
for i, b in enumerate(branches):
    real_roll, im_roll = b['rot']
    ax2.plot(t[t>=td], real_roll, color=colors[i%3], linestyle='-', label=f"Real Roll {i+1}")
    ax2.fill_between(t[t>=td], real_roll - im_roll, real_roll + im_roll, color=colors[i%3], alpha=0.2)

ax2.set_title("Rotational Divergence (Imaginary Phase Shading)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Roll Angle (Radians)")
plt.tight_layout()
plt.show()



​Unified Field Equation for Post-Divergence Trajectory Synthesis


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. setup time perimeters

t = np.linspace(0, 20, 200)  # Extended time for orbital feel
td = 10  # Divergence at mid-point
alpha = [0.4, 0.3, 0.3]  # Adjusted weights

# 2.  Simple Keplerian orbit approximation

a = 5.0  # semi-major axis
e = 0.1  # eccentricity
omega = 2 * np.pi / 10  # orbital frequency (period 10 units)
x0 = a * (np.cos(omega * t) - e)
y0 = a * np.sqrt(1 - e**2) * np.sin(omega * t)
z0 = 0.1 * np.sin(2 * omega * t)  # Small z for 3D

# Pre-divergence orientation (stable spin)
roll0 = 0.05 * t
pitch0 = 0.02 * t
yaw0 = 0.01 * t

# 3. Post-Divergence Complex Branches with Barycenter-like perturbations
def generate_branches(t, td, x0, y0, z0, roll0, pitch0, yaw0):
    mask = t >= td
    post_td = t[mask] - td
    branches = []
    
    for i in range(len(alpha)):
        # Translational: Base + chaotic perturbation (e.g., Jupiter-like pull)
        perturb = 0.5 * (i + 1) * np.sin(0.5 * post_td + i)  # Sinusoidal divergence
        xi = x0[mask] + perturb * post_td**0.5  # Square root growth for diffusion-like
        yi = y0[mask] + perturb * np.cos(post_td)
        zi = z0[mask] + 0.2 * i * post_td
        
        # Imaginary tendency: Exponential sensitivity
        im_pos = np.exp(0.3 * post_td) * (i + 1)
        
        # Rotational: Real + imaginary wobble
        roll_i = roll0[mask] + 0.1 * i * post_td
        im_roll = np.sin(post_td * (i+1)) * (post_td * 0.05)  # Bounded growth
        
        # Similarly for pitch and yaw
        pitch_i = pitch0[mask] + 0.05 * i * post_td
        im_pitch = np.cos(post_td * (i+1)) * (post_td * 0.05)
        yaw_i = yaw0[mask] + 0.03 * i * post_td
        im_yaw = np.sin(1.5 * post_td * (i+1)) * (post_td * 0.05)
        
        branches.append({
            'pos': (xi, yi, zi, im_pos),
     
