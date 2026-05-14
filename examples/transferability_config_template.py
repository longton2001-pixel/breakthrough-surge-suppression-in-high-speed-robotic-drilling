"""
Transferability configuration template.

Use this file as a checklist when adapting the SIDSO/EMR workflow to a
different robot, end-effector, workpiece material, or drilling condition. The
optimization algorithm does not need to change; the physical parameters and
experimental calibration data should be updated to match the new setup.
"""

import numpy as np


# 1. Robot kinematics: replace with the target robot's MDH or equivalent model.
# Format: [alpha(i-1), a(i-1), theta_offset(i), d(i)] in radians and meters.
MDH_PARAMS = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])

# 2. Joint limits: replace with the target robot's controller limits in radians.
JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]

# 3. Dynamics: replace each link's mass, center of mass, and inertia tensor.
LINKS = [
    {"m": 0.0, "c": np.zeros(3), "I": np.eye(3)}
    for _ in range(6)
]

# 4. Stiffness: identify or provide joint stiffness values for the target robot.
JOINT_STIFFNESS = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# 5. Tool transform: update for the installed spindle/tool-holder assembly.
TOOL_ROTATION_MATRIX = np.eye(3)
TOOL_CENTER_OFFSET = np.zeros(3)

# 6. Process/material inputs: update for the target workpiece and drilling test.
# Materials with different thrust force or chip-formation behavior should use
# recalibrated drilling force, feed velocity, and validation/sensitivity data.
DRILLING_FORCE = 0.0
FEED_VELOCITY = 0.0
WORKPIECE_MATERIAL = "replace_with_material_name"
