# [file name]: run_enhanced.py
"""
Enhanced TAMER implementation addressing all assignment questions
- Multiple environment support (Q1)
- Hybrid learning variants (Q2) 
- Extended feedback interfaces (Q3)
- Clear demonstration of reward vs feedback differences (Q4)
"""

import asyncio
import gymnasium as gym
from tamer.agent_enhanced import TamerEnhanced
from tamer.environment_manager import EnvironmentManager

async def main():
    print("=== Enhanced TAMER - Social Robotics Assignment ===")
    
    # Question 1: Multiple environment support
    env_manager = EnvironmentManager()
    available_envs = env_manager.list_compatible_environments()
    print("Available environments:", available_envs)
    
    # Choose environment (easily configurable for different environments)
    env_name = 'MountainCar-v0'  # Can be changed to 'CartPole-v1', etc.
    env = env_manager.create_environment(env_name, render_mode='rgb_array')
    
    # Question 2: Multiple learning modes
    learning_modes = ['tamer', 'q_learning', 'hybrid', 'transfer']
    
    # Configuration for different experiments
    config = {
        'mode': 'hybrid',  # Try different modes to demonstrate variants
        'num_episodes': 3,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'min_eps': 0.01,
        'ts_len': 0.4,
        'shaping_lambda': 0.7,  # For hybrid mode
        'transfer_alpha': 0.3,  # For transfer learning
        'feedback_interface': 'enhanced_keyboard'  # Question 3: Interface options
    }
    
    print(f"Training with: {env_name}, Mode: {config['mode']}")
    
    agent = TamerEnhanced(env, **config)
    
    # Training phase
    print("\n--- Starting Training Phase ---")
    await agent.train(model_file_to_save='enhanced_tamer')
    
    # Evaluation phase  
    print("\n--- Starting Evaluation Phase ---")
    agent.evaluate(n_episodes=10)
    
    # Demonstrate learning progress
    agent.analyze_performance()
    
    print("\n=== Experiment Complete ===")

if __name__ == '__main__':
    asyncio.run(main())