# Stiffness-inertia dynamic synergy for breakthrough surge suppression in high-speed robotic drilling

This repository contains the core algorithms and evaluation scripts for the paper:

**"Stiffness-inertia dynamic synergy for breakthrough surge suppression in high-speed robotic drilling"**  
*Authors: Dong Liu, Bohan Feng, Qunfei Gu, Shun Liu, Mian Li, Sun Jin*  
*Shanghai Jiao Tong University*

## Abstract
Driven by the shift toward large-scale integrated die-casting in new energy vehicle (NEV) manufacturing, industrial robots are being explored for high-speed drilling. However, the "breakthrough surge" triggered by the rapid release of stored elastic energy significantly affects machining accuracy. To tackle this, we propose a **Stiffness-Inertia Dynamic Synergistic Optimization (SIDSO)** strategy. We introduce an **Energy-Mass Ratio (EMR)** metric to quantify the combined effect of system compliance and operational-space inertia. A **Hybrid Multi-Start Sequential Quadratic Programming (HMS-SQP)** algorithm is used for global posture optimization. Machining tests show that SIDSO-optimized postures reduce the average surge velocity amplification ratio by 53.5% and the mean hole diameter error by 39.5%.

## Research Highlights
- Reveal the transient energy conversion mechanism of the breakthrough surge and establish an Energy-Mass Ratio metric for dynamic modeling.
- Propose a Stiffness-Inertia Dynamic Synergistic Optimization strategy to effectively suppress transient impacts during high-speed robotic drilling.
- Develop a Hybrid Multi-Start Sequential Quadratic Programming algorithm for global posture optimization in non-convex configuration spaces.

## Directory Structure

```text
core/             # Core modules: EMR optimization, kinematics, and stiffness identification
scripts/          # Reproduction, validation, sensitivity-analysis, and plotting scripts
data/             # Experimental datasets for stiffness, optimization, validation, kinematics, and damping sensitivity
results/          # Generated evaluation plots and sensitivity-analysis tables
examples/         # Adaptation templates for other robot and material configurations
tests/            # Lightweight regression tests
main.py           # Main entry point for the EMR-based pose optimization demo
requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/longton2001-pixel/breakthrough-surge-suppression-in-high-speed-robotic-drilling.git
   cd breakthrough-surge-suppression-in-high-speed-robotic-drilling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Robot Parameters

The kinematic, dynamic, and stiffness parameters for the FANUC R-2000iC robot are centrally managed in `core/config.py`.

### 1. Dynamics & Inertia
The inertial parameters (mass, center of mass, and inertia tensor) for each link were obtained from **official CAD assemblies**.
- **Component Details**: Each of the six link assemblies considers the complete physical structure. Specifically, the **high-speed electric spindle** and the drilling tool-holder are fully integrated into the mass and inertia properties of the **6th link (J6)** to ensure the accuracy of operational-space mass calculations.
- **Process**: Material properties (e.g., high-strength alloy steel, aluminum, or motor-specific densities) were assigned to each component in CAD. Mass properties were then measured relative to the defined MDH coordinate frames.
- **Accuracy**: These parameters represent the "as-built" configuration used in the drilling experiments.

### 2. Joint Stiffness
Joint stiffness values were identified using the experimental data and the identification algorithm provided in `core/stiffness_identification.py`.

## Transferability to Other Robots and Materials

The SIDSO framework separates the optimization algorithm from robot- and process-specific physical parameters. To transfer the method to another robot configuration, update the kinematic parameters, link inertial properties, joint limits, joint stiffness matrix, and tool-frame transform in `core/config.py`. The same HMS-SQP search and EMR objective can then be reused with the new robot model.

For a different workpiece material or drilling condition, the process inputs should be recalibrated because thrust force, feed velocity, stiffness response, and damping behavior may change with the material and cutting parameters. The template in `examples/transferability_config_template.py` summarizes the required robot, tool, and material/process fields that should be checked before applying the workflow to a new setup.

## Quick Start

### 1. Pose Optimization Demo
To see the EMR-based pose generation algorithm in action:
- `python main.py`: Run the core EMR pose optimization demo with baseline comparisons.

### 2. Reproduce Research Figures
Run the corresponding scripts to generate the figures presented in the paper:
- `python scripts/run_stiffness_identification.py`: Execute the joint stiffness identification process using experimental data in `data/stiffness_data/`.
- `python scripts/plot_kinematics.py`: Generate **Fig 14** (Kinematics analysis).
- `python scripts/plot_optimization.py`: Generate **Fig 12** (Optimization comparison).
- `python scripts/plot_validation_final.py`: Generate **Fig 9** (Validation results).
- `python scripts/plot_diameter_error.py`: Generate **Fig 13** (Diameter error analysis).

### 3. Damping Sensitivity Analysis
- `python scripts/damping_sensitivity_analysis.py`: Reproduce the damping sensitivity analysis using `data/damping_sensitivity_data.csv` and write `results/damping_sensitivity_table.csv`.

### 4. Tests
- `python -m unittest discover tests`: Run the lightweight regression tests.

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{yourcitation,
  title={...},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (Note: The core optimization engine is provided as source code here for review; please contact the authors for commercial use.)
