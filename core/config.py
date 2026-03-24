"""
Robot Configuration Center
Functional: Contains all physical parameters for the FANUC R-2000iC robot. 
Modify this file to adapt to different robot models.
Outputs: MDH_PARAMS, LINKS (dynamics), JOINT_STIFFNESS, JOINT_LIMITS, 
         TOOL_ROTATION_MATRIX, TOOL_CENTER_OFFSET, DRILLING_FORCE.
"""

import numpy as np

# ============================================================
# 1. Robot Kinematics (MDH Parameters)
# ============================================================
# Format: [alpha(i-1), a(i-1), theta(i), d(i)]
# Note: theta is the initial offset; actual calculation adds joint angle q.
# Units: Radians, Meters.
# Data based on R-2000iC general structure.
MDH_PARAMS = np.array([
    [0,       0,       0,       0.670     ], # Joint 1
    [-np.pi/2, 0.312,   -np.pi/2, 0     ], # Joint 2
    [0,       1.075,   0,       0     ], # Joint 3
    [-np.pi/2, 0.225,   0,       1.280], # Joint 4
    [np.pi/2,  0,       0,       0     ], # Joint 5
    [-np.pi/2, 0,       0,       0.215]  # Joint 6
])

# Joint Limits (Radians)
JOINT_LIMITS = [
    (-3.2, 3.2),  # J1
    (-1.0, 1.3),  # J2
    (-2.3, 3.0),  # J3
    (-6.2, 6.2),  # J4
    (-2.0, 2.0),  # J5
    (-6.2, 6.2)   # J6
]

# ============================================================
# 2. Dynamic Parameters
# ============================================================
# Derived from CAD/Identification (Units: kg, m, kg*m^2)
# Format: 
# m: mass
# c: center of mass (COM) relative to Frame i
# I: inertia tensor around COM
LINKS = [
    { # Link 1
        'm': 262.614,
        'c': np.array([-0.0139, 0.0765, -0.1107]),
        'I': np.array([
            [12.2537, 2.1602, 5.2151],
            [2.1602, 21.1149, -2.1459],
            [5.2151, -2.1459, 22.6950]
        ])
    },
    { # Link 2
        'm': 156.154,
        'c': np.array([0.4248, 0.04287, 0.2163]),
        'I': np.array([
            [2.0806, 0.6780, 2.1482],
            [0.6780, 19.4489, 0.0592],
            [2.1482, 0.0592, 19.9772]
        ])
    },
    { # Link 3
        'm': 149.341,
        'c': np.array([0.1342, 0.2415, 0.0391]),
        'I': np.array([
            [16.1975, 2.8773, -0.6570],
            [2.8773, 3.9591, -1.0987],
            [-0.6570, -1.0987, 16.9705]
        ])
    },
    { # Link 4
        'm': 10.190,
        'c': np.array([0.0000, 0.0019, -0.1055]),
        'I': np.array([
            [0.0977, 0.0000, 0.0000],
            [0.0000, 0.0968, 0.0080],
            [0.0000, 0.0080, 0.0281]
        ])
    },
    { # Link 5
        'm': 17.698,
        'c': np.array([0.0000, 0.0543, -0.0277]),
        'I': np.array([
            [0.2239, 0.0000, 0.0000],
            [0.0000, 0.1571, 0.0419],
            [0.0000, 0.0419, 0.1473]
        ])
    },
    { # Link 6
        'm': 41.450,
        'c': np.array([0.0005, -0.0336, 0.2654]),
        'I': np.array([
            [0.9806, -0.0006, 0.0021],
            [-0.0006, 0.8190, -0.0846],
            [0.0021, -0.0846, 0.3065]
        ])
    }
]

# ============================================================
# 3. Stiffness (Identified Results)
# ============================================================
# Joint stiffness matrix K_theta (Nm/rad)
JOINT_STIFFNESS = np.diag([
    9.339417645122125978e+05, # J1
    4.129256136278848164e+06, # J2
    1.907534091464656871e+06, # J3
    2.332046026475337567e+05, # J4
    4.043588275329603930e+05, # J5
    7.961815940385307476e+04  # J6
])

# External drilling force magnitude (N)
DRILLING_FORCE = 500

# ============================================================
# 4. Tool Mounting Parameters
# ============================================================
# Tool frame orientation relative to Flange (Frame 6)
# Drill points along the Y+ axis of the flange.
TOOL_ROTATION_MATRIX = np.array([
     [0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]
 ])

# Tool Center Point (TCP) offset relative to flange center (Meters)
TOOL_CENTER_OFFSET = np.array([0.0069, 0.2453, 0.3193])
