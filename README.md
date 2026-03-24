# Stiffness-inertia dynamic synergy for breakthrough surge suppression in high-speed robotic drilling

This repository contains the core algorithms and evaluation scripts for the paper: 

**"Stiffness-inertia dynamic synergy for breakthrough surge suppression in high-speed robotic drilling"**  
*Authors: Dong Liu, Bohan Feng, Sun Jin*  
*Shanghai Jiao Tong University*

## Abstract
Driven by the shift toward large-scale integrated die-casting in new energy vehicle (NEV) manufacturing, industrial robots are being explored for high-speed drilling. However, the "breakthrough surge" triggered by the rapid release of stored elastic energy significantly affects machining accuracy. To tackle this, we propose a **Stiffness-Inertia Dynamic Synergistic Optimization (SIDSO)** strategy. We introduce an **Energy-Mass Ratio (EMR)** metric to quantify the combined effect of system compliance and operational-space inertia. A **Hybrid Multi-Start Sequential Quadratic Programming (HMS-SQP)** algorithm is used for global posture optimization. Machining tests show that SIDSO-optimized postures reduce the average surge velocity amplification ratio by 53.5% and the mean hole diameter error by 39.5%.

## Research Highlights
- Reveal the transient energy conversion mechanism of the breakthrough surge and establish an Energy-Mass Ratio metric for dynamic modeling.
- Propose a Stiffness-Inertia Dynamic Synergistic Optimization strategy to effectively suppress transient impacts during high-speed robotic drilling.
- Develop a Hybrid Multi-Start Sequential Quadratic Programming algorithm for global posture optimization in non-convex configuration spaces.
- Demonstrate that the SIDSO strategy significantly reduces surge velocity amplification and hole diameter errors compared to single-objective methods.

## Directory Structure

```text
├── core/           # Core modules: EMR optimization (HMS-SQP), Kinematics, Stiffness Identification
├── scripts/        # Data validation and plotting scripts
├── data/           # Experimental datasets (Stiffness, Optimization, Kinematics)
├── results/        # Generated evaluation plots (Fig 9, 12, 13, 14)
├── main.py         # Main entry point for EMR-based pose optimization demo
└── requirements.txt
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

## Quick Start

### 1. Pose Optimization Demo
To see the EMR-based pose generation algorithm in action:
- `python main.py`: Run the core EMR pose optimization demo with baseline comparisons (DE/DA).

### 2. Reproduce Research Figures
Run the corresponding scripts to generate the figures presented in the paper:
- `python scripts/run_stiffness_identification.py`: Execute the joint stiffness identification process using experimental data in `data/stiffness_data/`.
- `python scripts/plot_kinematics.py`: Generate **Fig 14** (Kinematics analysis)
- `python scripts/plot_optimization.py`: Generate **Fig 12** (Optimization comparison)
- `python scripts/plot_validation_final.py`: Generate **Fig 9** (Validation results)
- `python scripts/plot_diameter_error.py`: Generate **Fig 13** (Diameter error analysis)

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
