# Learning-Augmented Control (LAC) 

This repository contains a comprehensive simulation framework for evaluating Learning-Augmented Control (LAC) against traditional control methods under various system conditions and adversarial attacks.

## Overview

The project implements and compares four control methods:
- **LAC (Learning-Augmented Control)**: Our proposed adaptive control method that learns confidence parameters with untrusted predictions
- **MPC (Model Predictive Control)**: Traditional MPC with predictions
- **LQR (Linear Quadratic Regulator)**: Nominal LQR control
- **Self-Tuning**: Adaptive control with self-tuning parameters

## System Types

The framework supports multiple system types:
- **Linear System**: 4-dimensional linear system with tracking objectives
- **Robotic Arm**: Single-joint robotic arm with nonlinear dynamics

## Main Simulation Scripts

### 1. `pipeline.py` - Robustness Analysis
**Purpose**: Evaluates control method robustness under varying prediction noise levels.

**Key Features**:
- Sweeps through different noise levels (μ from 0.1 to 5.0)
- Runs 5 trials per noise level for statistical significance
- Compares all four control methods
- Generates cost vs. prediction error plots with confidence intervals

**Output**: `linear.png` - Shows how control costs vary with prediction uncertainty

**Usage**:
```bash
python pipeline.py
```

### 2. `run_robotic_arms_seaborn.py` - Robotic Arm Analysis
**Purpose**: Comprehensive analysis of control methods on robotic arm system with adversarial attacks.

**Key Features**:
- Uses seaborn styling for publication-quality plots
- Implements adversarial attacks during simulation
- 2x2 subplot layout showing state, control, cost, and confidence
- Attack region highlighting and statistical analysis

**Output**: `robotic_arm_results_2col_fancy.pdf` - Publication-ready figure with 4 subplots

**Usage**:
```bash
python run_robotic_arms_seaborn.py
```

### 3. `run_one_simulation.py` - Linear System Attack Analysis
**Purpose**: Detailed analysis of control methods under adversarial attacks on linear systems.

**Key Features**:
- 3-panel figure showing disturbance prediction, controller performance, and confidence learning
- Attack phase analysis (before, during, after attack)
- Tracking trajectory comparison
- RMSE calculations for quantitative comparison

**Outputs**:
- `perturbation_attack_results.png` - Main 3-panel analysis
- `tracking_trajectory_comparison_attack_and_after.png` - Trajectory comparison

**Usage**:
```bash
python run_one_simulation.py
```

## Core Components

### Control Methods

#### LAC (Learning-Augmented Control)
- **Algorithm**: Delayed Confidence Learning (DCL)
- **Key Feature**: Dynamically adjusts confidence parameter λₜ ∈ [0,1]
- **Update Rule**: λₜ = λₜ₋ₖ - η∇ξₜ₋ₖ(λ)
- **Advantage**: Adapts quickly to system changes and attacks

#### MPC (Model Predictive Control)
- **Algorithm**: Standard MPC with prediction horizon N=5
- **Key Feature**: Uses predicted disturbances φ̂ₜ
- **Limitation**: Fixed model, no adaptation

#### LQR (Linear Quadratic Regulator)
- **Algorithm**: Nominal LQR control
- **Key Feature**: Uses κₜ = 0 (no disturbance prediction)
- **Limitation**: No adaptation to changing conditions

#### Self-Tuning
- **Algorithm**: Self-tuning control with FTL (Follow the Leader)
- **Key Feature**: Adaptive parameter estimation
- **Limitation**: Slower adaptation compared to LAC

### System Dynamics

#### Linear System (4D)
```
xₜ₊₁ = Axₜ + Buₜ + φₜ
```
where φₜ represents unknown disturbances/tracking objectives.

#### Robotic Arm (1D)
```
xₜ₊₁ = xₜ + α sin(xₜ) + βuₜ exp(-|xₜ|) + φₜ
```
where xₜ is the arm angle and φₜ represents external disturbances.

### Adversarial Attack Model

The framework implements realistic adversarial attacks:
- **Attack Period**: t ∈ [T/3, 2T/3]
- **Attack Pattern**: Periodic injection of large disturbances
- **Attack Magnitude**: 4× normal disturbance level
- **Attack Interval**: Every 5 time steps

## Dependencies

```python
numpy
matplotlib
scipy
seaborn
control  # for DARE solver
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LAC
```

2. Install dependencies:
```bash
pip install numpy matplotlib scipy seaborn control
```

## Usage Examples

### Basic Simulation
```python
from main import run_dynamics

# Run LAC on linear system
cost_lac, x_lac, u_lac, lambda_lac, phi_pred, phi_true = run_dynamics(
    noise_mu=0, sigma_eta=0, SYSTEM_TYPE='linear', METHOD_TYPE='lac', number_tests=1
)
```

### Parameter Configuration
Edit `_PARAMETERS.py` to modify:
- System matrices (A, B, Q, R)
- Attack parameters
- Control bounds
- Learning rates

## Key Results

### 1. Robustness to Prediction Noise
- LAC maintains lowest cost across all noise levels
- Traditional methods degrade significantly with high noise
- Self-tuning shows intermediate performance

### 2. Attack Resilience
- LAC recovers fastest after attack termination
- MPC shows good performance during attack but slower recovery
- LQR fails to adapt to attack conditions
- Self-tuning adapts but slower than LAC

### 3. Tracking Performance
- LAC achieves lowest RMSE during and after attacks
- Dynamic confidence adjustment enables quick adaptation
- Other methods either lack adaptation or adapt too slowly

## File Structure

```
LAC/
├── README.md                           # This file
├── _PARAMETERS.py                      # System parameters and configuration
├── main.py                            # Core simulation engine
├── dynamics.py                        # System dynamics definitions
├── generate_phi.py                    # Disturbance generation
├── self_tuning.py                     # Self-tuning control implementation
├── pipeline.py                        # Robustness analysis script
├── run_robotic_arms_seaborn.py        # Robotic arm analysis
├── run_one_simulation.py              # Linear system attack analysis
├── run_robotic_arms.py                # Basic robotic arm simulation
├── plot.py                           # Robotic arm visualization
└── robot_arm_visualization.py        # Advanced arm visualization
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lac2025,
  title={Learning-Augmented Control: Adaptively
Confidence Learning for Competitive MPC},
  author={Tongxin Li},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.