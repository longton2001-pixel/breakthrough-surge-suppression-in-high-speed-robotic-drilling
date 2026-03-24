"""
Joint Stiffness Identification Runner
Executes the full identification pipeline:
1. Load experimental data from data/stiffness_data/
2. Run robust identification (outlier rejection)
3. Report identified joint stiffness values (Nm/rad).
"""

import sys
import os
import numpy as np

# Add core directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from stiffness_identification import StiffnessIdentifier
from robot_model import RobotModel

def main():
    print("=" * 60)
    print(f"{'Robot Joint Stiffness Identification':^52}")
    print("=" * 60)

    # 1. Path Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'stiffness_data')
    
    csv_q = os.path.join(DATA_DIR, 'joint_angles.csv')
    csv_f = os.path.join(DATA_DIR, 'forces.csv')
    csv_dx = os.path.join(DATA_DIR, 'deltaX.csv')

    # File Check
    for f in [csv_q, csv_f, csv_dx]:
        if not os.path.exists(f):
            print(f"[Error] Data file not found: {f}")
            return

    # 2. Data Loading
    try:
        data_q = np.loadtxt(csv_q, delimiter=',', skiprows=1)
        data_f = np.loadtxt(csv_f, delimiter=',', skiprows=1)
        data_dx = np.loadtxt(csv_dx, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"[Error] Failed to read data: {e}")
        return

    num_samples = data_q.shape[0]
    print(f"Successfully loaded {num_samples} experimental points.")

    # 3. Process measurements
    identifier = StiffnessIdentifier()
    robot_helper = RobotModel() 

    print("Building observation matrix...")
    for i in range(num_samples):
        q_model = robot_helper.map_raw_to_model(data_q[i, :6])
        f_3d = data_f[i, :3]
        F_6d = np.zeros(6)
        F_6d[:3] = f_3d
        delta_X = data_dx[i, :3]
        identifier.add_measurement(q_model, F_6d, delta_X)

    # 4. Identification with Outlier Filtering
    print("Executing Robust Least Squares Identification...")
    c_est, k_est, report = identifier.solve_robust(
        std_threshold=1.5, 
        filter_mode='relative', 
        min_deformation=0.0002
    )

    # 5. Output Results
    if k_est is not None:
        print("\n" + "-" * 50)
        print(f"{'Joint':<10} | {'Identified Stiffness (Nm/rad)':<25}")
        print("-" * 50)
        for i in range(6):
            print(f"Joint {i+1:<4} | {k_est[i]:<25.2e}")
        print("-" * 50)
        
        # Statistics
        valid_count = sum(1 for item in report if not item['is_outlier'])
        print(f"\n[Stats] Valid points: {valid_count} / {num_samples} (Rejection: {(num_samples-valid_count)/num_samples*100:.1f}%)")
        
        # Save results for downstream tasks
        out_file = os.path.join(DATA_DIR, 'stiffness_result_reproduced.csv')
        np.savetxt(out_file, k_est, delimiter=",", header="Stiffness_Nm_per_rad")
        print(f"\nResult saved to: {out_file}")
    else:
        print("\n[Error] Identification process failed.")

if __name__ == "__main__":
    main()
