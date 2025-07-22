# utils.py

import numpy as np
from math import sin, cos, atan2, sqrt

def normalize_angle(angle):

    return atan2(sin(angle), cos(angle))

def jacobian_motion(x_r, u, dt):

    theta = x_r[2]
    v = u[0]
    omega = u[1]

    if abs(omega) < 1e-6:
        F_x = np.array([
            [1.0, 0.0, -v * dt * sin(theta)],
            [0.0, 1.0, v * dt * cos(theta)],
            [0.0, 0.0, 1.0]
        ])
    else:
        F_x = np.array([
            [1.0, 0.0, (-v/omega) * cos(theta) + (v/omega) * cos(theta + omega * dt)],
            [0.0, 1.0, (-v/omega) * sin(theta) + (v/omega) * sin(theta + omega * dt)],
            [0.0, 0.0, 1.0]
        ])
    return F_x

def motion_model(x_r, u, dt):

    x, y, theta = x_r
    v, omega = u

    if abs(omega) < 1e-6:
        x_new = x + v * dt * cos(theta)
        y_new = y + v * dt * sin(theta)
        theta_new = theta
    else:
        x_new = x + (v / omega) * (sin(theta + omega * dt) - sin(theta))
        y_new = y - (v / omega) * (cos(theta + omega * dt) - cos(theta))
        theta_new = normalize_angle(theta + omega * dt)
    return np.array([x_new, y_new, theta_new])

def jacobian_measurement(x_r, landmark_pos):

    rx, ry, rtheta = x_r
    lx, ly = landmark_pos

    dx = lx - rx
    dy = ly - ry
    q = dx**2 + dy**2
    sqrt_q = sqrt(q)

    Hr = np.array([
        [-dx/sqrt_q, -dy/sqrt_q, 0.0],
        [dy/q,       -dx/q,      -1.0]
    ])

    Hl = np.array([
        [dx/sqrt_q, dy/sqrt_q],
        [-dy/q,     dx/q]
    ])
    return Hr, Hl, q

def predict_measurement(x_r, landmark_pos):

    dx = landmark_pos[0] - x_r[0]
    dy = landmark_pos[1] - x_r[1]
    _range = sqrt(dx**2 + dy**2)
    _bearing = normalize_angle(atan2(dy, dx) - x_r[2])
    return np.array([_range, _bearing])

def calc_mahalanobis_distance(z, h, S):
    dz = z - h
    dz[1] = normalize_angle(dz[1])
    return dz.T @ np.linalg.inv(S) @ dz