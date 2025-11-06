# [file name]: config.py
"""
Configuration management for TAMER experiments
Centralized settings for different experimental setups
"""

# Experiment configurations for different assignment questions
EXPERIMENT_CONFIGS = {
    'q1_environment_exploration': {
        'description': 'Question 1 - Different environments',
        'environments': ['MountainCar-v0', 'CartPole-v1'],
        'mode': 'tamer',
        'num_episodes': 2,
        'ts_len': 0.4
    },
    'q2_algorithm_variants': {
        'description': 'Question 2 - Algorithm variants', 
        'environments': ['MountainCar-v0'],
        'modes': ['tamer', 'q_learning', 'hybrid', 'transfer'],
        'num_episodes': 3,
        'ts_len': 0.3
    },
    'q3_interface_comparison': {
        'description': 'Question 3 - Interface comparison',
        'environments': ['MountainCar-v0'],
        'mode': 'tamer',
        'interfaces': ['simple_keyboard', 'enhanced_keyboard'],
        'num_episodes': 2,
        'ts_len': 0.5
    },
    'q4_learning_analysis': {
        'description': 'Question 4 - Learning analysis',
        'environments': ['MountainCar-v0'],
        'modes': ['tamer', 'q_learning'],
        'num_episodes': 3,
        'ts_len': 0.4,
        'detailed_logging': True
    }
}

# Default configuration
DEFAULT_CONFIG = {
    'mode': 'hybrid',
    'num_episodes': 3,
    'discount_factor': 0.95,
    'epsilon': 0.1,
    'min_eps': 0.01,
    'ts_len': 0.4,
    'shaping_lambda': 0.7,
    'transfer_alpha': 0.3,
    'feedback_interface': 'enhanced_keyboard',
    'render_mode': 'rgb_array',
    'output_dir': 'logs'
}

def get_experiment_config(experiment_name):
    """Get configuration for specific experiment"""
    if experiment_name in EXPERIMENT_CONFIGS:
        config = DEFAULT_CONFIG.copy()
        config.update(EXPERIMENT_CONFIGS[experiment_name])
        return config
    else:
        return DEFAULT_CONFIG.copy()

def display_config_summary(config):
    """Display configuration summary"""
    print("\n=== Experiment Configuration ===")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print("="*40)