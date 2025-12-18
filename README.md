​Unified Field Equation for Post-Divergence Trajectory Synthesis


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Setup Time and Parameters
t = np.linspace(0, 20, 200)  # Extended time for orbital feel
td = 10  # Divergence at mid-point
alpha = [0.4, 0.3, 0.3]  # Adjusted weights

# 2. Pre-Divergence Path (Elliptical orbit around 'sun')
# Simple Keplerian orbit approximation
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
     
