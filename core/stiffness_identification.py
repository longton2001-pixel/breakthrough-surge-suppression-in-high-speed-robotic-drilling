"""
Joint Stiffness Identification
Provides the StiffnessIdentifier class for identifying joint stiffness parameters
based on cable-driven force-deformation measurements.
"""

import numpy as np
from robot_model import RobotModel

class StiffnessIdentifier:
    def __init__(self):
        self.robot = RobotModel()
        self.A_list = []  # Observation matrix blocks
        self.Y_list = []  # Measurement vector blocks
        self.num_samples = 0

    def add_measurement(self, q, F_cable, delta_X_diff):
        """Add a single differential measurement point."""
        J = self.robot.compute_jacobian(q)
        tau_cable = J.T @ F_cable
        A_row = np.zeros((3, 6))
        for i in range(6):
            A_row[:, i] = J[:3, i] * tau_cable[i]
        self.A_list.append(A_row)
        self.Y_list.append(delta_X_diff)
        self.num_samples += 1

    def solve(self):
        """Execute Least Squares to solve for compliance and stiffness."""
        if self.num_samples < 6:
            print("Insufficient data point (minimum 6 required).")
            return None, None
        
        Phi = np.vstack(self.A_list)
        Y = np.hstack(self.Y_list) 
        rank = np.linalg.matrix_rank(Phi)
        cond = np.linalg.cond(Phi)

        if rank < 6:
            print("Observation matrix is rank deficient.")
            return None, None

        c_compliance, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
        k_stiffness = np.array([1.0/val if abs(val) > 1e-12 else 0.0 for val in c_compliance])
        return c_compliance, k_stiffness
    
    def solve_robust(self, std_threshold=2.0, filter_mode='relative', min_deformation=0.2):
        """
        Execute robust stiffness identification with outlier filtering.
        - std_threshold: Multiple of standard deviation for filtering.
        - filter_mode: 'absolute' or 'relative' (based on deformation).
        """
        N = self.num_samples
        if N < 6:
            print("Insufficient data.")
            return None, None, None

        Phi_all = np.vstack(self.A_list)
        Y_all = np.hstack(self.Y_list)
        deformation_norms = np.linalg.norm(np.array(self.Y_list), axis=1)

        # Initial solution
        c_init, _, _, _ = np.linalg.lstsq(Phi_all, Y_all, rcond=None)
        abs_err_norms = np.linalg.norm((Y_all - Phi_all @ c_init).reshape(N, 3), axis=1)
        
        if filter_mode == 'relative':
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = abs_err_norms / (deformation_norms + 1e-9)
            
            valid_stats_indices = deformation_norms > min_deformation
            stats_data = rel_errors[valid_stats_indices] if np.sum(valid_stats_indices) > 5 else rel_errors
            limit_rel = np.mean(stats_data) + std_threshold * np.std(stats_data)
            
            valid_indices = []
            outlier_indices = []
            for i in range(N):
                is_outlier = (abs_err_norms[i] > 0.0001) if deformation_norms[i] < min_deformation else (rel_errors[i] > limit_rel)
                if is_outlier: outlier_indices.append(i)
                else: valid_indices.append(i)
            valid_indices, outlier_indices = np.array(valid_indices), np.array(outlier_indices)
        else:
            limit = np.mean(abs_err_norms) + std_threshold * np.std(abs_err_norms)
            valid_indices = np.where(abs_err_norms < limit)[0]
            outlier_indices = np.where(abs_err_norms >= limit)[0]

        if len(valid_indices) < 6:
            c_final = c_init
        else:
            Phi_valid = np.vstack([self.A_list[i] for i in valid_indices])
            Y_valid = np.hstack([self.Y_list[i] for i in valid_indices])
            c_final, _, _, _ = np.linalg.lstsq(Phi_valid, Y_valid, rcond=None)

        # Final Report
        Y_pred_final = (Phi_all @ c_final).reshape(N, 3)
        final_abs_err = np.linalg.norm(np.array(self.Y_list) - Y_pred_final, axis=1)
        k_stiffness = np.array([1.0/val if abs(val) > 1e-12 else 0.0 for val in c_final])

        report = []
        for i in range(N):
            report.append({
                "id": i + 1,
                "error_mm": final_abs_err[i] * 1000,
                "deformation_mm": deformation_norms[i] * 1000,
                "relative_error": (final_abs_err[i] / (deformation_norms[i] + 1e-9) * 100) if deformation_norms[i] > 1e-6 else 0.0,
                "is_outlier": i in outlier_indices,
                "dx_measured": self.Y_list[i]
            })
        return c_final, k_stiffness, report