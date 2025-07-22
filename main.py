import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from robot import Robot
from ekf_slam import EKF_SLAM
from utils import normalize_angle
from math import sqrt, atan2, sin, cos

DT = 0.1
SIM_TIME = 50.0

Q_SIM_MOTION = np.diag([0.2, 0.2, np.deg2rad(1.0)])**2
Q_ODOM_EKF = np.diag([0.1, 0.1, np.deg2rad(0.5)])**2

R_SENSOR_TRUE = np.diag([0.1, np.deg2rad(1.0)])**2
R_EKF_SENSOR = np.diag([0.1, np.deg2rad(1.0)])**2

MAX_LANDMARK_DETECT_DIST = 10.0
ASSOCIATION_THRESHOLD_PERCENTILE = 0.99

MAP_LANDMARKS = np.array([
    [5.0, 5.0],
    [0.0, 10.0],
    [-5.0, 5.0],
    [-5.0, 0.0],
    [-5.0, -5.0],
    [0.0, -10.0],
    [5.0, -5.0],
    [10.0, 0.0],
    [7.0, 7.0]
])

initial_robot_pose = np.array([0.0, 0.0, 0.0])
u_sim_control = np.array([1.0, 0.1])

robot_sim = Robot(initial_robot_pose, Q_SIM_MOTION, R_SENSOR_TRUE, MAX_LANDMARK_DETECT_DIST)
ekf_slam = EKF_SLAM(initial_robot_pose, Q_ODOM_EKF, R_EKF_SENSOR, ASSOCIATION_THRESHOLD_PERCENTILE)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.grid(True)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("EKF SLAM Simulation")

ax.plot(MAP_LANDMARKS[:, 0], MAP_LANDMARKS[:, 1], "sk", markersize=6, label="True Landmarks")

history_true_pose = []
history_est_robot_pose = []

time = 0.0
while time < SIM_TIME:
    time += DT

    u_odom_input_to_ekf = robot_sim.move(u_sim_control, DT)
    
    history_true_pose.append(robot_sim.true_pose.copy())

    ekf_slam.predict(u_odom_input_to_ekf, DT)

    sim_measurements = robot_sim.get_measurements(MAP_LANDMARKS)

    ekf_slam.update(sim_measurements)

    history_est_robot_pose.append(ekf_slam.x_est[0:3].copy())

    ax.clear()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"EKF SLAM Simulation (Time: {time:.1f}s)")

 
    hist_true_arr = np.array(history_true_pose)
    ax.plot(hist_true_arr[:, 0], hist_true_arr[:, 1], "-g", label="True Robot Path")
    ax.plot(robot_sim.true_pose[0], robot_sim.true_pose[1], "og", markersize=8, label="True Robot")

    hist_est_arr = np.array(history_est_robot_pose)
    ax.plot(hist_est_arr[:, 0], hist_est_arr[:, 1], "--b", label="Estimated Robot Path")
    ax.plot(ekf_slam.x_est[0], ekf_slam.x_est[1], "ob", markersize=8, label="Estimated Robot")
    
    if ekf_slam.P_est.shape[0] >= 2:
        P_robot_xy = ekf_slam.P_est[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eig(P_robot_xy)
        if all(e > 0 for e in eigvals):
            ellipse_angle = atan2(eigvecs[1, 0], eigvecs[0, 0])

            width = sqrt(eigvals[0]) * 2 * 2.4477
            height = sqrt(eigvals[1]) * 2 * 2.4477
            ellipse = Ellipse(xy=(ekf_slam.x_est[0], ekf_slam.x_est[1]),
                              width=width, height=height,
                              angle=np.degrees(ellipse_angle), edgecolor='blue', fc='None', lw=1.5, alpha=0.7)
            ax.add_patch(ellipse)

    for i in range(ekf_slam.known_landmarks):
        lm_idx = 3 + 2 * i
        lm_x = ekf_slam.x_est[lm_idx]
        lm_y = ekf_slam.x_est[lm_idx + 1]
        ax.plot(lm_x, lm_y, "+r", markersize=10, label="Estimated Landmark" if i == 0 else "")

        lm_cov = ekf_slam.P_est[lm_idx : lm_idx + 2, lm_idx : lm_idx + 2]
        if all(e > 0 for e in np.linalg.eigvalsh(lm_cov)):
            eigvals_lm, eigvecs_lm = np.linalg.eig(lm_cov)
            ellipse_angle_lm = atan2(eigvecs_lm[1, 0], eigvecs_lm[0, 0])
            width_lm = sqrt(eigvals_lm[0]) * 2 * 2.4477
            height_lm = sqrt(eigvals_lm[1]) * 2 * 2.4477
            ellipse_lm = Ellipse(xy=(lm_x, lm_y), width=width_lm, height=height_lm,
                                 angle=np.degrees(ellipse_angle_lm), edgecolor='red', fc='None', lw=1, linestyle='--', alpha=0.6)
            ax.add_patch(ellipse_lm)

    ax.plot(MAP_LANDMARKS[:, 0], MAP_LANDMARKS[:, 1], "sk", markersize=6, label="True Landmarks")

    ax.legend(loc='upper right')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    plt.pause(0.01)

plt.show()

print("\n--- Simulation Complete ---")
print(f"Final Estimated Robot Pose: {ekf_slam.x_est[0:3]}")
print(f"True Robot Pose: {robot_sim.true_pose}")
print(f"Number of landmarks estimated: {ekf_slam.known_landmarks}")