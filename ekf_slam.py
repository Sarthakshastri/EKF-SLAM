
import numpy as np
from utils import normalize_angle, jacobian_motion, motion_model, jacobian_measurement, predict_measurement, calc_mahalanobis_distance
from scipy.stats import chi2
from math import sin, cos, sqrt

class EKF_SLAM:
    def __init__(self, initial_robot_pose, Q_odom_ekf, R_ekf_sensor, association_threshold):

        self.x_est = np.array(initial_robot_pose).astype(float)
        
        self.P_est = np.diag([0.1, 0.1, np.deg2rad(1.0)])**2

        self.Q_odom_ekf = Q_odom_ekf
        self.R_ekf_sensor = R_ekf_sensor
        self.association_threshold = chi2.ppf(association_threshold, df=2)

        self.known_landmarks = 0
        self.landmark_id_map = {}

    def predict(self, u_odom, dt):

        self.x_est[0:3] = motion_model(self.x_est[0:3], u_odom, dt)

        G = jacobian_motion(self.x_est[0:3], u_odom, dt)

        self.P_est[0:3, 0:3] = G @ self.P_est[0:3, 0:3] @ G.T + self.Q_odom_ekf

        if self.known_landmarks > 0:
            self.P_est[0:3, 3:] = G @ self.P_est[0:3, 3:]
            self.P_est[3:, 0:3] = self.P_est[3:, 0:3] @ G.T

    def update(self, measurements):

        for z_obs, true_land_id in measurements:

            min_md = float('inf')
            best_match_idx_in_state = -1

            for i in range(self.known_landmarks):
                land_idx_in_state = 3 + 2 * i
                estimated_landmark_pos = self.x_est[land_idx_in_state : land_idx_in_state + 2]

                h_pred = predict_measurement(self.x_est[0:3], estimated_landmark_pos)
                Hr, Hl, q_val = jacobian_measurement(self.x_est[0:3], estimated_landmark_pos)

                H_i = np.zeros((2, len(self.x_est)))
                H_i[:, 0:3] = Hr
                H_i[:, land_idx_in_state : land_idx_in_state + 2] = Hl

                S_i = H_i @ self.P_est @ H_i.T + self.R_ekf_sensor

                md = calc_mahalanobis_distance(z_obs, h_pred, S_i)

                if md < min_md:
                    min_md = md
                    best_match_idx_in_state = land_idx_in_state

            if min_md < self.association_threshold and best_match_idx_in_state != -1:

                current_landmark_idx_in_state = best_match_idx_in_state
                
                estimated_landmark_pos = self.x_est[current_landmark_idx_in_state : current_landmark_idx_in_state + 2]
                h_pred = predict_measurement(self.x_est[0:3], estimated_landmark_pos)
                
                Hr, Hl, q_val = jacobian_measurement(self.x_est[0:3], estimated_landmark_pos)

                H = np.zeros((2, len(self.x_est)))
                H[:, 0:3] = Hr
                H[:, current_landmark_idx_in_state : current_landmark_idx_in_state + 2] = Hl

                S = H @ self.P_est @ H.T + self.R_ekf_sensor
                K = self.P_est @ H.T @ np.linalg.inv(S)

                innovation = z_obs - h_pred
                innovation[1] = normalize_angle(innovation[1])
                self.x_est = self.x_est + K @ innovation
                self.P_est = (np.eye(len(self.x_est)) - K @ H) @ self.P_est

            else:

                new_landmark_pos_x = self.x_est[0] + z_obs[0] * cos(self.x_est[2] + z_obs[1])
                new_landmark_pos_y = self.x_est[1] + z_obs[0] * sin(self.x_est[2] + z_obs[1])
                
                self.x_est = np.append(self.x_est, [new_landmark_pos_x, new_landmark_pos_y])

                old_dim = self.P_est.shape[0]
                new_dim = old_dim + 2
                
                P_new = np.zeros((new_dim, new_dim))
                P_new[0:old_dim, 0:old_dim] = self.P_est
                
                range_val = z_obs[0]
                bearing_val = z_obs[1]
                theta_r = self.x_est[2]
                
                Jx_init = np.array([
                    [1.0, 0.0, -range_val * sin(theta_r + bearing_val)],
                    [0.0, 1.0, range_val * cos(theta_r + bearing_val)]
                ])
                
                Jz_init = np.array([
                    [cos(theta_r + bearing_val), -range_val * sin(theta_r + bearing_val)],
                    [sin(theta_r + bearing_val), range_val * cos(theta_r + bearing_val)]
                ])
                
                P_rl_new = Jx_init @ self.P_est[0:3, 0:3]
                P_new[old_dim:, 0:3] = P_rl_new
                P_new[0:3, old_dim:] = P_rl_new.T

                P_ll_new = Jx_init @ self.P_est[0:3, 0:3] @ Jx_init.T + Jz_init @ self.R_ekf_sensor @ Jz_init.T
                P_new[old_dim:, old_dim:] = P_ll_new

                if self.known_landmarks > 0:
                    P_existing_lm_robot = self.P_est[3:old_dim, 0:3]
                    P_new[3:old_dim, old_dim:] = P_existing_lm_robot @ Jx_init.T
                    P_new[old_dim:, 3:old_dim] = (P_existing_lm_robot @ Jx_init.T).T
                
                self.P_est = P_new
                self.known_landmarks += 1
 
                self.landmark_id_map[true_land_id] = old_dim