# [file name]: tamer/environment_manager.py
"""
Environment manager addressing Question 1
- Centralized environment handling
- Multiple environment support
- Environment compatibility checking
"""

import gymnasium as gym

class EnvironmentManager:
    """
    Manages different Gym environments for TAMER experimentation
    Addresses Question 1: Multiple environment support
    """
    
    def __init__(self):
        self.supported_environments = {
            'MountainCar-v0': {
                'type': 'classic_control',
                'action_space': 'discrete',
                'observation_space': 'continuous',
                'suitable_for_tamer': True,
                'description': 'Car must climb hill with limited power'
            },
            'CartPole-v1': {
                'type': 'classic_control', 
                'action_space': 'discrete',
                'observation_space': 'continuous',
                'suitable_for_tamer': True,
                'description': 'Balance pole on moving cart'
            },
            'MountainCarContinuous-v0': {
                'type': 'classic_control',
                'action_space': 'continuous',
                'observation_space': 'continuous', 
                'suitable_for_tamer': False,
                'description': 'Continuous version of MountainCar'
            },
            'Pendulum-v1': {
                'type': 'classic_control',
                'action_space': 'continuous',
                'observation_space': 'continuous',
                'suitable_for_tamer': False,
                'description': 'Swing up and balance pendulum'
            }
        }
    
    def list_compatible_environments(self):
        """List environments suitable for TAMER"""
        compatible = []
        for env_name, info in self.supported_environments.items():
            if info['suitable_for_tamer']:
                compatible.append({
                    'name': env_name,
                    'description': info['description'],
                    'action_space': info['action_space']
                })
        return compatible
    
    def create_environment(self, env_name, render_mode='rgb_array'):
        """
        Create and return a Gym environment
        Handles environment-specific configurations
        """
        if env_name not in self.supported_environments:
            raise ValueError(f"Environment {env_name} not supported or tested")
            
        env_info = self.supported_environments[env_name]
        
        if not env_info['suitable_for_tamer']:
            print(f"Warning: {env_name} may not be ideal for TAMER due to {env_info['action_space']} action space")
        
        try:
            env = gym.make(env_name, render_mode=render_mode)
            print(f"Created environment: {env_name}")
            print(f"  Action Space: {env.action_space}")
            print(f"  Observation Space: {env.observation_space}")
            return env
        except Exception as e:
            print(f"Error creating environment {env_name}: {e}")
            raise
    
    def get_environment_info(self, env_name):
        """Get detailed information about an environment"""
        if env_name in self.supported_environments:
            return self.supported_environments[env_name]
        else:
            return None
    
    def test_environment_compatibility(self, env_name):
        """Test if environment is compatible with current TAMER implementation"""
        if env_name not in self.supported_environments:
            return False, "Environment not in supported list"
            
        env_info = self.supported_environments[env_name]
        
        if env_info['action_space'] != 'discrete':
            return False, "TAMER currently requires discrete action spaces"
            
        return True, "Environment is compatible"