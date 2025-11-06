# TAMER+: Enhanced Human-in-the-Loop Reinforcement Learning

This repository extends the original [TAMER](https://github.com/benibienz/TAMER) framework for the "Social Robotics" course at Sorbonne Université. TAMER (Training an Agent Manually via Evaluative Reinforcement) enables human-guided reinforcement learning where human feedback directly shapes agent behavior.

## Enhanced Features

### 🚀 Multiple Learning Modes
- **Pure TAMER**: Original human feedback-only learning
- **Q-Learning**: Traditional environment reward learning  
- **Hybrid**: Combined human + environment rewards
- **Transfer**: Adaptive model weighting between human and environment signals

### 🎮 Enhanced Interaction
- **Multi-level feedback**: 6-level keyboard input (vs original binary)
- **Improved visualization**: Real-time action and feedback display
- **Multiple environments**: Support for MountainCar, CartPole, and more
- **Advanced logging**: Detailed performance analysis and statistics

### 🔧 Technical Improvements
- **Modular architecture**: Separated environment, algorithm, and interface components
- **Configuration management**: Pre-set experiments for different research questions
- **Comprehensive compatibility**: Updated dependencies and cross-environment support

## Quick Start

### Basic Usage (Original TAMER)

```bash
python run.py
```

Watch the agent and press 'W' for positive feedback, 'A' for negative feedback.

### Enhanced Experiments

```bash
python run_enhanced.py
```

Explore multiple learning modes and environment configurations.

## Project Structure

```bash
TAMER-enhanced/
├── run_enhanced.py              # Enhanced experiment runner
├── run.py                       # Original TAMER implementation
├── tamer/
│   ├── agent_enhanced.py        # Multi-algorithm TAMER variants
│   ├── interface_enhanced.py    # Advanced feedback interfaces
│   ├── environment_manager.py   # Multi-environment support
│   ├── config.py               # Experiment configurations
│   ├── agent.py                # Original TAMER agent
│   └── interface.py            # Original interface
└── requirements.txt
```

## Requirements

- Python 3.7+
- numpy, sklearn, pygame, gymnasium, opencv-python

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Research Applications

This enhanced implementation supports investigation of:

- Human-robot interaction paradigms
- Algorithm comparisons (TAMER vs Q-learning vs hybrid)
- Multi-modal feedback interfaces
- Environment compatibility analysis
- Learning efficiency studies

Key Changes from Original:

- Updated dependencies: gym → gymnasium, modern package versions
- Enhanced rendering: OpenCV for consistent visualization
- Extended functionality: Multiple algorithms and interfaces
- Research-ready: Comprehensive logging and analysis tools
- Modular design: Easy extension and experimentation

## Documentation

See the included Jupyter notebook report for detailed analysis of algorithm performance, interface design, and theoretical foundations of human-in-the-loop reinforcement learning.

___

*Based on the original TAMER implementation by Knox & Stone (2009)*

