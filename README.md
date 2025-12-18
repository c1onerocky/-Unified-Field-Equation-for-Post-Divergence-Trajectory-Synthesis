import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters - tuned for clarity
t = np.linspace(0, 20, 400)
td = 8.0
num_branches = 40  # Much more readable
np.random.seed(42)
alphas = np.random.dirichlet(np.ones(num_branches))  # Some branches dominate

# Pre-divergence orbit
a, e = 6.0, 0.3
omega = 2 * np.pi / 12
x0 = a * (np.cos(omega * t) - e)
y0 = a * np.sqrt(1 - e**2) * np.sin(omega * t)
z0 = 0.5 * np.sin(0.5 * omega * t)

# Orientation
roll0 = 0.05 * t
pitch0 = 0.02 * t
yaw0 = 0.01 * t

mask_pre = t < td
mask_post = t >= td
post_t = t[mask_post] - td
div_idx = np.where(mask_post)[0][0]
P_d = np.array([x0[div_idx], y0[div_idx], z0[div_idx]])
theta_d = np.array([roll0[div_idx], pitch0[div_idx], yaw0[div_idx]]) + 1.0

# Generate clearer branches
branches = []
for i in range(num_branches):
    alpha = alphas[i]
    pert = 0.04 * np.random.randn(3)
    div_factor = np.exp(0.3 * post_t)  # Slower divergence = more structure visible
    
    X_i = pert[0] * div_factor + 1.2 * np.sin(post_t * 0.8 + i * 0.2)
    Y_i = pert[1] * div_factor + 0.8 * np.cos(post_t * 1.1 + i)
    Z_i = pert[2] * div_factor + 0.4 * (-1)**i * post_t**0.6
    
    V_complex = np.array([
        X_i + 1j * (-Y_i),
        Y_i + 1j * (-X_i),
        Z_i + 1j * Z_i
    ])
    
    roll_i = 0.06 * post_t
    pitch_i = 0.04 * post_t
    yaw_i = 0.02 * post_t
    delta_roll = 0.3 * np.sin(1.5 * post_t)
    delta_pitch = 0.25 * np.cos(post_t)
    delta_yaw = 0.2 * np.sin(2 * post_t)
    
    Theta_complex = np.array([
        roll_i + 1j * delta_roll,
        pitch_i + 1j * delta_pitch,
        yaw_i + 1j * delta_yaw
    ])
    
    im_rad = 8 * div_factor * alpha
    branches.append((V_complex, Theta_complex, im_rad, alpha))

# Superposition
S_c = np.zeros((3, len(post_t)), dtype=complex)
epsilon = 0.5
for V_c, Theta_c, _, alpha in branches:
    for comp in range(3):
        num = P_d[comp] + V_c[comp]
        den = theta_d[comp] + Theta_c[comp] + epsilon
        S_c[comp] += alpha * (num / den)

pos_super = np.real(S_c)
im_super = np.abs(np.imag(S_c)).mean(axis=0)  # Average latent divergence

# Plotting
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.set_pane_color((0,0,0,1))
ax.yaxis.set_pane_color((0,0,0,1))
ax.zaxis.set_pane_color((0,0,0,1))

# Pre-divergence
ax.plot(x0[mask_pre], y0[mask_pre], z0[mask_pre], 'w--', linewidth=4, label="Stable Pre-Divergence Orbit")

# Branches - colored by probability (hot = likely)
cmap = plt.cm.hot_r  # Red = high alpha, black = low
for idx, (V_c, _, im_rad, alpha) in enumerate(branches):
    xi = P_d[0] + np.real(V_c[0])
    yi = P_d[1] + np.real(V_c[1])
    zi = P_d[2] + np.real(V_c[2])
    color = cmap(alpha**0.5)  # Exaggerate high-prob branches
    lw = 0.5 + 4 * alpha
    alphaline = 0.3 + 0.7 * alpha
    ax.plot(xi, yi, zi, color=color, linewidth=lw, alpha=alphaline)

# Superposed path - bright magenta with uncertainty envelope
ax.plot(pos_super[0], pos_super[1], pos_super[2], 'm-', linewidth=6, label="Most Probable Path (Re(S_c))", alpha=0.9)

# Optional: faint envelope from imaginary part
envelope_alpha = 0.15
ax.plot(pos_super[0] + im_super*2, pos_super[1], pos_super[2], 'm-', alpha=envelope_alpha)
ax.plot(pos_super[0] - im_super*2, pos_super[1], pos_super[2], 'm-', alpha=envelope_alpha)
ax.plot(pos_super[0], pos_super[1] + im_super*2, pos_super[2], 'm-', alpha=envelope_alpha)
ax.plot(pos_super[0], pos_super[1] - im_super*2, pos_super[2], 'm-', alpha=envelope_alpha)

ax.set_xlim(-8, 12)
ax.set_ylim(-8, 12)
ax.set_zlim(-4, 8)
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')
ax.set_title('Post-Divergence Trajectory Synthesis\n(Probabilistically Weighted Branches)', color='white', fontsize=16)
ax.legend(facecolor='black', labelcolor='white', loc='upper left')
ax.grid(True, color='gray', alpha=0.3)
ax.view_init(elev=28, azim=-70)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
