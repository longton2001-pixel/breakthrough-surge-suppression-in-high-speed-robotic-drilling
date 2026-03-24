"""
Robotic Drilling Pose Optimization Demo
Main entry point for comparing HMS-SQP with baseline global optimization algorithms.
Evaluates four typical configurations (EMR Best, Stiff Best, Mass Best, EMR Worst)
and reports performance metrics.
"""

import sys
import os
import time
import numpy as np
from scipy.optimize import minimize

# Add core directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from robot_model import RobotModel
from optimizer import DrillingOptimizer
from benchmarks import Benchmarks
import config

def main():
    # 1. Initialization
    robot = RobotModel()
    optimizer = DrillingOptimizer(robot)
    tester = Benchmarks(robot, optimizer)
    
    print("=" * 80)
    print(f"{'Robotic Drilling Pose Optimization (HMS-SQP vs Baselines)':^80}")
    print("=" * 80)

    # 2. Define Task Goal (Based on a typical teach pendant pose)
    input_q_raw_deg = np.array([-97.91, 31.218, -24.525, 179.968, 65.487, -307.7]) 
    input_q_model = robot.map_raw_to_model(np.radians(input_q_raw_deg))
    T_target = robot.get_end_effector_pose(input_q_model)
    target_pos = T_target[:3, 3]
    target_n = T_target[:3, 2] # Target drilling direction
    target_n = target_n / np.linalg.norm(target_n)

    print(f"[Target Task]")
    print(f"  Position (m) : {target_pos}")
    print(f"  Normal Vector: {target_n}\n")

    # 3. Search for Four Typical Configurations
    print("Searching for typical comparative configurations...")
    
    # (1) EMR Global Best
    print("  [1/4] Searching for EMR-Optimal configuration...", end=" ", flush=True)
    res_emr = optimizer.optimize(target_pos, target_n)
    print("Done")

    # (2) Stiffness Best
    print("  [2/4] Searching for Stiffness-Maximized configuration...", end=" ", flush=True)
    def cost_stiffness_only(q, n):
        _, _, Ct, _ = robot.get_matrices_for_cost(q)
        compliance = (n.T @ Ct[:3, :3] @ n) 
        penalty = 100.0 * optimizer._smooth_limit_penalty(q) + 10000.0 * optimizer._robust_singularity_penalty(robot.compute_jacobian(q))
        return compliance + penalty
    
    res_stiff = minimize(cost_stiffness_only, x0=res_emr.x, args=(target_n,), method='SLSQP', bounds=config.JOINT_LIMITS, 
                         constraints=({'type': 'eq', 'fun': lambda q: optimizer.constraint_position(q, target_pos)},
                                      {'type': 'eq', 'fun': lambda q: optimizer.constraint_orientation(q, target_n)}))
    print("Done")

    # (3) Mass Best
    print("  [3/4] Searching for Mass-Maximized configuration...", end=" ", flush=True)
    def cost_mass_only(q, n):
        J, _, _, Lambda_inv_full = robot.get_matrices_for_cost(q)
        Lambda_full = np.linalg.inv(Lambda_inv_full + 1e-6*np.eye(6))
        m_eff = n.T @ Lambda_full[:3,:3] @ n
        penalty = 100.0 * optimizer._smooth_limit_penalty(q) + 10000.0 * optimizer._robust_singularity_penalty(J)
        return -m_eff + penalty
    res_mass = minimize(cost_mass_only, x0=res_emr.x, args=(target_n,), method='SLSQP', bounds=config.JOINT_LIMITS,
                        constraints=({'type': 'eq', 'fun': lambda q: optimizer.constraint_position(q, target_pos)},
                                     {'type': 'eq', 'fun': lambda q: optimizer.constraint_orientation(q, target_n)}))
    print("Done")

    # (4) EMR Worst Case (Valid but Bad)
    print("  [4/4] Searching for Performance-Worst (Valid) configuration...", end=" ", flush=True)
    bad_q = None
    max_cost = -1.0
    for _ in range(50):
        seed = [np.random.uniform(b[0], b[1]) for b in config.JOINT_LIMITS]
        try:
            q_ik = robot.compute_numerical_ik(target_pos, target_n, initial_q=seed)
            if optimizer.constraint_position(q_ik, target_pos) < 1e-3:
                details = optimizer._compute_cost_components(q_ik, target_n)
                if details['limit_penalty'] < 0.1 and details['cost_opt'] > max_cost:
                    max_cost = details['cost_opt']
                    bad_q = q_ik
        except: continue
    print("Done")

    # 4. Generate Comparison Report
    print("\n" + "=" * 80)
    print(f"{'Performance Comparison: Four Typical Postures':^80}")
    print("=" * 80)
    qs = [res_emr.x, res_stiff.x, res_mass.x, bad_q]
    
    header = f"{'Metric':<20} | {'EMR Best':^13} | {'Stiff Best':^13} | {'Mass Best':^13} | {'EMR Worst':^13}"
    print(header)
    print("-" * 80)
    
    metrics = []
    for q in qs:
        if q is not None:
            metrics.append(optimizer._compute_cost_components(q, target_n))
        else:
            metrics.append(None)
            
    # Row printing helper
    def print_row(name, key, fmt, scale=1.0, inv=False):
        row = f"{name:<20}"
        for m in metrics:
            if m:
                val = 1.0/m[key] if inv else m[key]
                row += f" | {fmt.format(val * scale):^13}"
            else:
                row += f" | {'N/A':^13}"
        print(row)
    
    print_row("Axial Stiff. (N/m)", "comp_axial", "{:.1e}", inv=True)
    print_row("Equiv. Mass (kg)", "m_eff", "{:.2f}")
    print_row("Physical Cost", "cost_opt", "{:.4f}")
    print("-" * 80)
    
    # 5. Algorithm Performance Benchmarking
    print(f"\n{'Algorithm Performance Benchmarking (Global Search)':^80}")
    print("-" * 80)
    print(f"{'Algorithm':<25} | {'Cost (1/Stiff)':^15} | {'Time (s)':^15}")
    print("-" * 80)
    
    # HMS-SQP (Ours)
    t0 = time.time()
    res_sqp = optimizer.optimize(target_pos, target_n)
    t_sqp = time.time() - t0
    c_sqp = optimizer._compute_cost_components(res_sqp.x, target_n)['cost_opt']
    print(f"{'HMS-SQP (Ours)':<25} | {c_sqp:^15.4f} | {t_sqp:^15.4f}")
    
    # DE
    t0 = time.time()
    res_de = tester.run_de(target_pos, target_n)
    t_de = time.time() - t0
    c_de = optimizer._compute_cost_components(res_de.x, target_n)['cost_opt']
    print(f"{'Differential Evolution':<25} | {c_de:^15.4f} | {t_de:^15.4f}")
    
    # DA
    t0 = time.time()
    res_da = tester.run_da(target_pos, target_n)
    t_da = time.time() - t0
    c_da = optimizer._compute_cost_components(res_da.x, target_n)['cost_opt']
    print(f"{'Dual Annealing':<25} | {c_da:^15.4f} | {t_da:^15.4f}")

    print("-" * 80)
    print("\n[Notes]")
    print("1. Typical configurations demonstrate the trade-off balanced by the EMR algorithm.")
    print("2. Benchmarks demonstrate HMS-SQP's superior convergence speed for the drilling task.")

if __name__ == "__main__":
    main()