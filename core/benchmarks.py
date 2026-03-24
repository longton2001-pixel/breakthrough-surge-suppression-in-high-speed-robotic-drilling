"""
Optimization Benchmarks
Provides wrappers for global optimization algorithms to compare against HMS-SQP.
Supports: Differential Evolution (DE), Dual Annealing (DA), Basin Hopping (BH).
"""

import sys
import os
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
import config

class Benchmarks:
    def __init__(self, robot, drilling_optimizer):
        self.robot = robot
        self.opt = drilling_optimizer
        self.bounds = config.JOINT_LIMITS
        self.penalty_weight = 1e6

    def _penalty_objective(self, q, target_pos, target_n):
        """Convert inequality/equality constraints into a single penalty-based objective."""
        try:
            components = self.opt._compute_cost_components(q, target_n)
            phys_cost = components["total_cost"]
        except:
            return 1e10
            
        pos_err = self.opt.constraint_position(q, target_pos)
        rot_err = self.opt.constraint_orientation(q, target_n)
        return phys_cost + self.penalty_weight * (pos_err + rot_err**2)

    def run_de(self, target_pos, target_n, max_fevals=2000):
        """Run Differential Evolution algorithm."""
        return differential_evolution(
            self._penalty_objective, bounds=self.bounds,
            args=(target_pos, target_n),
            maxiter=max_fevals // 50,
            popsize=15, polish=True
        )

    def run_da(self, target_pos, target_n, max_fevals=2000):
        """Run Dual Annealing algorithm."""
        return dual_annealing(
            self._penalty_objective, bounds=self.bounds,
            args=(target_pos, target_n),
            maxiter=max_fevals // 2
        )

    def run_bh(self, target_pos, target_n, niter=100):
        """Run Basin Hopping algorithm."""
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds}
        return basinhopping(
            self._penalty_objective, x0,
            niter=niter, minimizer_kwargs=minimizer_kwargs,
            args=(target_pos, target_n)
        )
