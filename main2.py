import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_in = pd.read_csv('./data/victoria_park.csv', index_col=0)
data_in.index = pd.to_timedelta(data_in.index, unit='s')
data_in = data_in.resample('1s').mean()
data_in = data_in.dropna()

# Constants
L = 2.83 
dt = 1.0
a = 0.5 
b = 3.78 

laser_cols = [col for col in data_in.columns if 'laser' in col]
laser_angles = np.linspace(-np.pi / 2, np.pi / 2, len(laser_cols))

# Initialize state
mu = np.array([0.0, 0.0, 0.0])  # x, y, yaw
sigma = np.diag([1.0, 1.0, 0.1])  # initial covariance
trajectory = [mu[:2].copy()]
landmarks = []

# Motion model
def fx(mu, v, delta, dt=1.0):
    mu_new = mu.copy()
    x, y, theta = mu[:3]
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt
    mu_new[0:3] = [x, y, theta]
    return mu_new

# Jacobian
def F_fx(mu, v, delta, dt=1.0):
    """Jacobian of the motion model for the full state."""
    n = len(mu)
    _, _, theta = mu[:3]

    F = np.eye(n)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] =  v * np.cos(theta) * dt
    return F

# Motion noise
Q = np.diag([0.5, 0.5, 0.05])**2

for _, row in data_in.iterrows():
    v = row['speed']
    delta = row['steering']

    # Predict
    mu = fx(mu, v, delta, dt)
    F = F_fx(mu, v, delta, dt)
    Q_aug = np.zeros_like(sigma)
    Q_aug[:3, :3] = Q  # motion noise only applies to robot pose
    sigma = F @ sigma @ F.T + Q_aug

    # Extract valid laser measurements
    ranges = row[laser_cols].values
    valid = (ranges > 1.0) & (ranges < 80.0)
    ranges = ranges[valid]
    angles = laser_angles[valid]

    # Compute camera position in world frame
    x, y, theta = mu[:3]
    cam_x = x + a * np.cos(theta) - b * np.sin(theta)
    cam_y = y + a * np.sin(theta) + b * np.cos(theta)

    for r, angle in zip(ranges, angles):
        lx = cam_x + r * np.cos(theta + angle)
        ly = cam_y + r * np.sin(theta + angle)

        MIN_DIST = 5.99  # Chi-square 95% threshold for 2 DOF
        best_j = -1
        min_dist = float('inf')

        for j in range((len(mu) - 3) // 2):
            lm_idx = 3 + 2 * j
            lx, ly = mu[lm_idx:lm_idx + 2]
            delta_lm = np.array([lx - x, ly - y])
            q = delta_lm @ delta_lm
            z_hat = np.array([np.sqrt(q), np.arctan2(delta_lm[1], delta_lm[0]) - theta])
            z_hat[1] = (z_hat[1] + np.pi) % (2 * np.pi) - np.pi

            sqrt_q = np.sqrt(q)
            dx, dy = delta_lm
            H = np.zeros((2, len(mu)))
            H[0, 0] = -dx / sqrt_q
            H[0, 1] = -dy / sqrt_q
            H[0, lm_idx] = dx / sqrt_q
            H[0, lm_idx + 1] = dy / sqrt_q
            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1
            H[1, lm_idx] = -dy / q
            H[1, lm_idx + 1] = dx / q

            S = H @ sigma @ H.T + np.diag([0.5, 0.1])**2
            dz = np.array([r, angle]) - z_hat
            dz[1] = (dz[1] + np.pi) % (2 * np.pi) - np.pi

            d2 = dz.T @ np.linalg.inv(S) @ dz

            if d2 < min_dist:
                min_dist = d2
                best_j = j
                best_H = H
                best_S = S
                best_dz = dz

        if min_dist < MIN_DIST:
            K = sigma @ best_H.T @ np.linalg.inv(best_S)
            mu = mu + K @ best_dz
            sigma = (np.eye(len(mu)) - K @ best_H) @ sigma
        else:
            mu = np.append(mu, [lx, ly])
            n = len(mu)
            sigma = np.pad(sigma, ((0, 2), (0, 2)), mode='constant')
            sigma[n-2:, n-2:] = np.eye(2) * 1e3  # High uncertainty for new landmark


    trajectory.append(mu[:2].copy())

# Convert to arrays
trajectory = np.array(trajectory)
landmarks = np.array(landmarks)

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Estimated Path')

if landmarks.shape[0] > 0 and landmarks.ndim == 2:
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='red', alpha=0.5, label='Landmarks')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('EKF SLAM - Victoria Park')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
plt.savefig('ekf_slam_victoria_park.png', dpi=300)
plt.close()