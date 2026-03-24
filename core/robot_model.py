"""
Core Algorithm Library (Robot Model)
Provides the RobotModel class for 6-DOF kinematics and dynamics computations.
Input: Joint angle vector q (6,), supports Raw (Teach Pendant) or Model (MDH) formats.
Output: Forward/Inverse Kinematics, Jacobian, Mass Matrix, Compliance Matrix, 
        Operational Space Inverse Inertia, and Numeric IK solver.
"""

import numpy as np
import config

class RobotModel:
    def __init__(self):
        """Initialize with parameters from config.py"""
        self.mdh = config.MDH_PARAMS
        self.links = config.LINKS
        self.K_inv = np.linalg.inv(config.JOINT_STIFFNESS)
        self.num_joints = 6
        self.R_tool_fixed = config.TOOL_ROTATION_MATRIX
        self.tcp_offset = config.TOOL_CENTER_OFFSET

    def map_raw_to_model(self, raw_q):
        """
        Map raw joint angles from the controller to MDH model angles.
        Handles typical parallelogram linkage decoupling.
        """
        model_q = np.zeros(6)
        model_q[0] = raw_q[0]
        model_q[1] = raw_q[1]
        model_q[2] = -(raw_q[1] + raw_q[2])
        model_q[3] = -raw_q[3]
        model_q[4] = -raw_q[4]
        model_q[5] = -raw_q[5]
        return model_q

    def map_model_to_raw(self, model_q):
        """
        Inverse map MDH model angles back to controller-executable raw angles.
        """
        raw_q = np.zeros(6)
        raw_q[0] = model_q[0]
        raw_q[1] = model_q[1]
        raw_q[2] = -model_q[2] - model_q[1] 
        raw_q[3] = -model_q[3]
        raw_q[4] = -model_q[4]
        raw_q[5] = -model_q[5]
        return raw_q

    def _mdh_transform(self, alpha, a, theta, d):
        """Compute single MDH transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,    -st,    0,      a],
            [st*ca, ct*ca, -sa,    -sa*d],
            [st*sa, ct*sa,  ca,     ca*d],
            [0,     0,      0,      1]
        ])

    def forward_kinematics_all(self, q):
        """Compute all link poses (T_0_i) for recursive dynamics."""
        transforms = []
        T_curr = np.eye(4)
        for i in range(self.num_joints):
            alpha, a, offset, d = self.mdh[i]
            T_i = self._mdh_transform(alpha, a, q[i] + offset, d)
            T_curr = T_curr @ T_i
            transforms.append(T_curr)
        return transforms

    def get_end_effector_pose(self, q):
        """Get Tool Tip pose in world coordinates."""
        T_flange = self.forward_kinematics_all(q)[-1]
        R_flange = T_flange[:3, :3]
        p_flange = T_flange[:3, 3]
        R_tool_world = R_flange @ self.R_tool_fixed
        p_tool_world = p_flange + R_flange @ self.tcp_offset
        T_tool = np.eye(4)
        T_tool[:3, :3] = R_tool_world
        T_tool[:3, 3] = p_tool_world
        return T_tool

    def compute_jacobian(self, q):
        """Compute 6x6 Geometric Jacobian at Tool Tip."""
        T_all = self.forward_kinematics_all(q)
        T_flange = T_all[-1]
        R_flange = T_flange[:3, :3]
        p_flange = T_flange[:3, 3]
        p_e = p_flange + R_flange @ self.tcp_offset 
        J = np.zeros((6, 6))
        for i in range(self.num_joints):
            T_i = T_all[i]
            z_i = T_i[:3, 2]  # Z-axis of current frame
            p_i = T_i[:3, 3]  # Origin of current frame
            J[:3, i] = np.cross(z_i, (p_e - p_i))
            J[3:, i] = z_i
        return J

    def compute_mass_matrix(self, q):
        """Compute 6x6 Joint-space Inertia Matrix M(q)."""
        M_q = np.zeros((6, 6))
        T_all = self.forward_kinematics_all(q)        
        for i in range(self.num_joints):
            link_data = self.links[i]
            mass = link_data['m']
            if mass == 0: continue
            T_0_i = T_all[i]
            R_0_i = T_0_i[:3, :3]
            p_0_i = T_0_i[:3, 3]
            p_com_world = p_0_i + R_0_i @ link_data['c']
            J_Li = np.zeros((6, 6))
            for j in range(i + 1):
                T_j = T_all[j]
                z_j = T_j[:3, 2]
                p_j = T_j[:3, 3]
                J_Li[:3, j] = np.cross(z_j, (p_com_world - p_j))
                J_Li[3:, j] = z_j
            I_world = R_0_i @ link_data['I'] @ R_0_i.T
            M_spatial_i = np.zeros((6, 6))
            M_spatial_i[:3, :3] = mass * np.eye(3)
            M_spatial_i[3:, 3:] = I_world
            M_q += J_Li.T @ M_spatial_i @ J_Li
        return M_q

    def get_matrices_for_cost(self, q):
        """Returns Jacobian, Mass Matrix, Compliance, and Inverse Inertia Lambda."""
        J = self.compute_jacobian(q)
        M = self.compute_mass_matrix(q)
        Ct = J @ self.K_inv @ J.T
        M_inv = np.linalg.pinv(M) 
        Lambda_inv_full = J @ M_inv @ J.T
        return J, M, Ct, Lambda_inv_full
    
    def get_drilling_direction(self, q):
        """Compute the current drilling vector n (3x1)."""
        T_tool = self.get_end_effector_pose(q)
        return T_tool[:3, 2] # Tool Z-axis

    def compute_numerical_ik(self, target_pos, target_n, initial_q=None, max_iter=100, tol_pos=1e-4, tol_rot=1e-3):
        """
        Robust Numerical Inverse Kinematics.
        target_pos: [x, y, z] (Meters)
        target_n: [nx, ny, nz] (Unit vector)
        """
        if initial_q is not None:
            q = np.array(initial_q, dtype=np.float64)
        else:
            j1_guess = np.arctan2(target_pos[1], target_pos[0])
            q = np.array([j1_guess, 0.1, 0.1, 0.0, 0.1, 0.0]) 
        
        n_des = np.array(target_n) / np.linalg.norm(target_n)
        alpha = 0.5 
        lambda_sq = 0.01**2 

        for k in range(max_iter):
            T_current = self.get_end_effector_pose(q)
            p_current = T_current[:3, 3]
            n_current = T_current[:3, 2]
            err_pos = target_pos - p_current
            err_rot = np.cross(n_current, n_des)
            pos_error_norm = np.linalg.norm(err_pos)
            rot_error_norm = np.linalg.norm(err_rot)
            if pos_error_norm < tol_pos and rot_error_norm < tol_rot:
                return q
            error = np.hstack([err_pos, err_rot])
            J = self.compute_jacobian(q)
            JJT = J @ J.T
            damping = lambda_sq * np.eye(6)
            grad = J.T @ np.linalg.solve(JJT + damping, error)
            max_step = 0.5 
            if np.linalg.norm(grad) > max_step:
                grad = grad / np.linalg.norm(grad) * max_step
            q = q + alpha * grad
            lower_limits = np.array([limit[0] for limit in config.JOINT_LIMITS])
            upper_limits = np.array([limit[1] for limit in config.JOINT_LIMITS])
            q = np.clip(q, lower_limits, upper_limits)
        return q