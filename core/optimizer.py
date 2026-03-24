"""
Drilling Optimizer
Finds the optimal joint configuration that minimizes end-effector surge velocity
under position and orientation constraints.
Algorithm: Hybrid Multi-Start Sequential Quadratic Programming (HMS-SQP)
Uses SLSQP with a mix of structured and random sampling for global search.
"""

import numpy as np
import config
from scipy.optimize import minimize

class DrillingOptimizer:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.force = config.DRILLING_FORCE
        
        # Optimization weights
        self.w_perf = 1.0       # Performance weight (EMR)
        self.w_limit = 1000.0   # Joint limit penalty weight
        self.w_sing = 1000.0    # Singularity penalty weight
        
        # Barrier margin for joint limits (0.087 rad approx 5 deg)
        self.limit_margin = 0.087

    def _smooth_limit_penalty(self, q):
        """
        Smooth joint limit penalty (C1 continuous quadratic barrier).
        Provides better gradients for the solver compared to piecewise functions.
        """
        limit_penalty = 0.0
        bounds = config.JOINT_LIMITS
        for i, val in enumerate(q):
            min_l, max_l = bounds[i]
            d_min = val - min_l
            d_max = max_l - val
            
            if d_min < self.limit_margin:
                limit_penalty += np.maximum(0, self.limit_margin - d_min)**2
                if d_min < 0: limit_penalty += 1000.0 * (d_min**2)

            if d_max < self.limit_margin:
                limit_penalty += np.maximum(0, self.limit_margin - d_max)**2
                if d_max < 0: limit_penalty += 1000.0 * (d_max**2)
        return limit_penalty

    def _robust_singularity_penalty(self, J, sigma_th=0.05):
        """
        Singularity penalty based on the minimum singular value.
        Penalizes configurations where the robot loses a degree of freedom.
        """
        try:
            s = np.linalg.svd(J, compute_uv=False)
            min_sigma = s[-1]
        except np.linalg.LinAlgError:
            return 1e6
        
        if min_sigma < sigma_th:
            return (sigma_th - min_sigma)**2
        return 0.0
    
    def _compute_cost_components(self, q, target_n):
        """Compute all cost components: Performance + Penalties."""
        J, M, Ct, Lambda_inv_full = self.robot.get_matrices_for_cost(q)
        
        # Part A: Physical Performance Cost (EMR-based)
        try:
            Lambda_full = np.linalg.inv(Lambda_inv_full)
        except np.linalg.LinAlgError:
            Lambda_full = np.linalg.inv(Lambda_inv_full + 1e-6 * np.eye(6))
        
        Lambda_vv = Lambda_full[:3, :3]
        comp_axial = target_n.T @ Ct[:3, :3] @ target_n
        m_eff = target_n.T @ Lambda_vv @ target_n
        
        if m_eff < 1e-4: m_eff = 1e-4
        cost_opt = (self.force**2) * (comp_axial / m_eff)        
        
        # Part B: Constraint Penalties
        limit_penalty = self._smooth_limit_penalty(q)
        singularity_penalty = self._robust_singularity_penalty(J)
        
        total_cost = (self.w_perf * cost_opt + self.w_limit * limit_penalty + self.w_sing * singularity_penalty)
        return {
            "total_cost": total_cost,
            "cost_opt": cost_opt,      
            "limit_penalty": limit_penalty,
            "singularity_penalty": singularity_penalty,
            "m_eff": m_eff,
            "comp_axial": comp_axial
        }

    def cost_function(self, q, target_n):
        """Objective function interface for the optimizer."""
        return self._compute_cost_components(q, target_n)["total_cost"]

    def constraint_position(self, q, target_pos):
        """Equality constraint for tool tip position."""
        current_pos = self.robot.get_end_effector_pose(q)[:3, 3]
        diff = current_pos - target_pos
        return np.dot(diff, diff)

    def constraint_orientation(self, q, target_n):
        """Equality constraint for tool orientation."""
        current_n = self.robot.get_drilling_direction(q)
        return 1.0 - np.dot(current_n, target_n)

    def optimize(self, target_pos, target_vec, initial_q=None, return_all=False):
        """
        Perform optimization using HMS-SQP.
        Returns the best result or a list of all converged solutions.
        """
        target_n = np.array(target_vec) / np.linalg.norm(target_vec)
        bounds = config.JOINT_LIMITS
        cons = (
            {'type': 'eq', 'fun': lambda q: self.constraint_position(q, target_pos)},
            {'type': 'eq', 'fun': lambda q: self.constraint_orientation(q, target_n)}
        )
        
        # Seed generation: Structured seeds for multiple configurations
        j1_base = np.arctan2(target_pos[1], target_pos[0])
        ELBOW, WRIST = [+1, -1], [+1, -1]
        seeds = []
        for e in ELBOW:
            for w in WRIST:
                seeds.append(np.array([j1_base, e * 1.0, -e * 1.5, 0.0, w * 1.2, 0.0]))
        
        # Global Search: Uniform random sampling
        for _ in range(100):
            seeds.append(np.array([np.random.uniform(b[0], b[1]) for b in bounds]))

        all_solutions = []
        for i, x0 in enumerate(seeds):
            try:
                res = minimize(fun=self.cost_function, x0=x0, args=(target_n,), method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-6, 'disp': False, 'maxiter': 500})
                if res.success:
                    pos_err = self.constraint_position(res.x, target_pos)
                    rot_err = self.constraint_orientation(res.x, target_n)
                    if pos_err < 1e-5 and rot_err < 1e-4:
                        details = self._compute_cost_components(res.x, target_n)
                        all_solutions.append({
                            "seed_id": i, "q": res.x, "res_object": res, "cost": details["total_cost"],
                            "m_eff": details["m_eff"], "pos_err": pos_err, "rot_err": rot_err,
                            "cost_opt": details["cost_opt"], "limit_penalty": details["limit_penalty"],
                            "singularity_penalty": details["singularity_penalty"]
                        })
            except: continue

        all_solutions.sort(key=lambda x: x['cost'])
        if return_all: return all_solutions
        return all_solutions[0]['res_object'] if all_solutions else None