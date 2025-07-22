# robot.py

import numpy as np
from utils import motion_model, predict_measurement, normalize_angle
from math import sqrt

class Robot:
    def __init__(self, initial_pose, Q_sim_motion, R_sim_sensor, max_landmark_dist):

        self.true_pose = np.array(initial_pose).astype(float)
        self.Q_sim_motion = Q_sim_motion
        self.R_sim_sensor = R_sim_sensor
        self.max_landmark_dist = max_landmark_dist

    def move(self, u_odom, dt):
        noisy_u_for_true_motion = u_odom + np.random.multivariate_normal([0, 0, 0], self.Q_sim_motion[:3,:3])[:2]
        self.true_pose = motion_model(self.true_pose, noisy_u_for_true_motion, dt)
        return u_odom

    def get_measurements(self, map_landmarks):

        sim_measurements = []
        for i, landmark_true_pos in enumerate(map_landmarks):
            dist = sqrt((self.true_pose[0] - landmark_true_pos[0])**2 + (self.true_pose[1] - landmark_true_pos[1])**2)
            if dist <= self.max_landmark_dist:
                true_measurement = predict_measurement(self.true_pose, landmark_true_pos)

                noisy_measurement = true_measurement + np.random.multivariate_normal([0, 0], self.R_sim_sensor)
                sim_measurements.append((noisy_measurement, i))
        return sim_measurements