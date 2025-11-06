# [file name]: tamer/agent_enhanced.py
"""
Enhanced TAMER agent addressing Questions 2 & 4
- Implements multiple learning variants
- Clear separation between reward-based and feedback-based learning
"""

import datetime as dt
import os
import pickle
import time
import uuid
import numpy as np
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter
import json

from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')
import cv2

class EnhancedSGDFunctionApproximator:
    """Enhanced function approximator with monitoring"""
    
    def __init__(self, env, model_type="generic"):
        self.model_type = model_type
        self.update_history = []
        
        # Feature preprocessing
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(5000)], dtype='float64'
        )

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # RBF feature union for better state representation
        self.featurizer = pipeline.FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=50)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=50)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=50)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=50)),
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # Initialize models for each action
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant', eta0=0.01)
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if action is None:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, target):
        """Question 4: Clear update mechanism showing supervised vs TD learning"""
        features = self.featurize_state(state)
        prediction_before = self.models[action].predict([features])[0]
        
        # Perform update
        self.models[action].partial_fit([features], [target])
        
        # Track update for analysis
        update_info = {
            'timestamp': time.time(),
            'prediction_before': prediction_before,
            'target': target,
            'prediction_after': self.models[action].predict([features])[0],
            'error': abs(target - prediction_before)
        }
        self.update_history.append(update_info)
        
        return update_info['error']

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]

class TamerEnhanced:
    """
    Enhanced TAMER agent implementing multiple variants from Question 2
    and clearly demonstrating differences from Question 4
    """
    
    def __init__(
        self,
        env,
        mode='tamer',           # 'tamer', 'q_learning', 'hybrid', 'transfer'
        num_episodes=2,
        discount_factor=0.95,
        epsilon=0.1,
        min_eps=0.01,
        ts_len=0.3,
        shaping_lambda=0.7,     # For hybrid mode
        transfer_alpha=0.3,     # For transfer learning
        feedback_interface='enhanced_keyboard',
        output_dir=LOGS_DIR,
        model_file_to_load=None
    ):
        self.mode = mode
        self.env = env
        self.ts_len = ts_len
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.shaping_lambda = shaping_lambda
        self.transfer_alpha = transfer_alpha
        self.feedback_interface_type = feedback_interface
        
        # Question 4: Clear model initialization based on learning type
        if model_file_to_load:
            self.load_model(model_file_to_load)
        else:
            self._initialize_models()
        
        # Hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = 0 if mode == 'tamer' else epsilon  # No exploration in pure TAMER
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / max(num_episodes, 1)
        
        # Enhanced logging for analysis
        self.setup_enhanced_logging()
        self.performance_history = []

    def _initialize_models(self):
        """Question 2: Initialize different model combinations based on mode"""
        if self.mode == 'tamer':
            # Pure human feedback learning
            self.H = EnhancedSGDFunctionApproximator(self.env, "Human_Model")
            print("Mode: TAMER (Human feedback only)")
            
        elif self.mode == 'q_learning':
            # Pure environment reward learning  
            self.Q = EnhancedSGDFunctionApproximator(self.env, "Q_Model")
            print("Mode: Q-learning (Environment reward only)")
            
        elif self.mode == 'hybrid':
            # Question 2: Hybrid approach - both models
            self.H = EnhancedSGDFunctionApproximator(self.env, "Human_Model")
            self.Q = EnhancedSGDFunctionApproximator(self.env, "Q_Model")
            print("Mode: Hybrid (Human feedback + Environment reward)")
            
        elif self.mode == 'transfer':
            # Question 2: Transfer learning between models
            self.H = EnhancedSGDFunctionApproximator(self.env, "Human_Model") 
            self.Q = EnhancedSGDFunctionApproximator(self.env, "Q_Model")
            print("Mode: Transfer learning")

    def setup_enhanced_logging(self):
        """Enhanced logging to demonstrate learning differences"""
        self.log_columns = [
            'Episode', 'Timestep', 'State', 'Action',
            'Human_Reward', 'Environment_Reward', 'Learning_Mode',
            'Update_Type', 'TD_Target', 'Model_Error'
        ]
        self.log_path = os.path.join(self.output_dir, f'{self.uuid}_detailed.csv')
        
        # Statistics file for analysis
        self.stats_path = os.path.join(self.output_dir, f'{self.uuid}_analysis.json')

    def act(self, state):
        """Enhanced action selection supporting all modes"""
        if np.random.random() < 1 - self.epsilon:
            if self.mode == 'transfer' and hasattr(self, 'Q') and hasattr(self, 'H'):
                # Question 2: Transfer learning - combine predictions
                q_preds = self.Q.predict(state)
                h_preds = self.H.predict(state)
                combined = [
                    (1 - self.transfer_alpha) * q + self.transfer_alpha * h 
                    for q, h in zip(q_preds, h_preds)
                ]
                return np.argmax(combined)
            elif hasattr(self, 'Q'):
                return np.argmax(self.Q.predict(state))
            else:
                return np.argmax(self.H.predict(state))
        else:
            return self.env.action_space.sample()

    def _train_episode(self, episode_index, disp):
        """Enhanced training episode with mode-specific learning"""
        print(f'\n--- Episode {episode_index + 1} ---')
        cv2.namedWindow('TAMER Enhanced Training', cv2.WINDOW_NORMAL)

        total_reward = 0
        state, _ = self.env.reset()
        
        with open(self.log_path, 'a+', newline='') as log_file:
            writer = DictWriter(log_file, fieldnames=self.log_columns)
            if episode_index == 0:
                writer.writeheader()
            
            for timestep in count():
                # Rendering
                frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('TAMER Enhanced Training', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # Action selection
                action = self.act(state)
                if disp:
                    disp.show_action(action)

                # Environment step
                next_state, env_reward, done, info, _ = self.env.step(action)
                
                # Question 4: Mode-specific learning updates
                learning_data = self._mode_specific_learning(
                    state, action, env_reward, next_state, done, disp, timestep
                )
                
                # Enhanced logging
                log_entry = {
                    'Episode': episode_index + 1,
                    'Timestep': timestep,
                    'State': str(state),
                    'Action': action,
                    'Environment_Reward': env_reward,
                    'Learning_Mode': self.mode
                }
                log_entry.update(learning_data)
                writer.writerow(log_entry)
                
                total_reward += env_reward
                state = next_state
                
                if done:
                    episode_stats = {
                        'episode': episode_index + 1,
                        'total_reward': total_reward,
                        'steps': timestep,
                        'mode': self.mode
                    }
                    self.performance_history.append(episode_stats)
                    print(f'Completed: Reward={total_reward}, Steps={timestep}')
                    break

        # Epsilon decay
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
            
        self._save_analysis_data()

    def _mode_specific_learning(self, state, action, env_reward, next_state, done, disp, timestep):
        """Question 4: Clear demonstration of different learning updates"""
        learning_data = {'Human_Reward': 0, 'Update_Type': '', 'TD_Target': 0, 'Model_Error': 0}
        
        if self.mode == 'tamer':
            # Pure human feedback learning
            human_reward = self._collect_human_feedback(disp)
            if human_reward != 0:
                error = self.H.update(state, action, human_reward)
                learning_data.update({
                    'Human_Reward': human_reward,
                    'Update_Type': 'Supervised_Human_Feedback',
                    'Model_Error': error
                })
                
        elif self.mode == 'q_learning':
            # Pure environment reward learning (TD learning)
            td_target = self._compute_td_target(state, action, next_state, env_reward, done)
            error = self.Q.update(state, action, td_target)
            learning_data.update({
                'Update_Type': 'TD_Environment_Reward',
                'TD_Target': td_target,
                'Model_Error': error
            })
            
        elif self.mode == 'hybrid':
            # Question 2: Hybrid learning - both updates
            human_reward = self._collect_human_feedback(disp)
            
            # Environment reward learning
            td_target = self._compute_td_target(state, action, next_state, env_reward, done)
            q_error = self.Q.update(state, action, td_target)
            
            # Human feedback learning
            h_error = 0
            if human_reward != 0:
                h_error = self.H.update(state, action, human_reward)
                
            learning_data.update({
                'Human_Reward': human_reward,
                'Update_Type': 'Hybrid_TD_Supervised',
                'TD_Target': td_target,
                'Model_Error': (q_error + h_error) / 2
            })
            
        elif self.mode == 'transfer':
            # Question 2: Transfer learning
            human_reward = self._collect_human_feedback(disp)
            
            # Always update Q with environment
            td_target = self._compute_td_target(state, action, next_state, env_reward, done)
            q_error = self.Q.update(state, action, td_target)
            
            # Update H with human feedback when available
            h_error = 0
            if human_reward != 0:
                h_error = self.H.update(state, action, human_reward)
                
            learning_data.update({
                'Human_Reward': human_reward,
                'Update_Type': 'Transfer_Learning',
                'TD_Target': td_target,
                'Model_Error': (q_error + h_error) / 2
            })
            
        return learning_data

    def _compute_td_target(self, state, action, next_state, reward, done):
        """Question 4: TD learning update rule"""
        if done and hasattr(next_state, '__len__') and len(next_state) > 0 and next_state[0] >= 0.5:
            return reward  # Terminal state
        else:
            next_q = np.max(self.Q.predict(next_state))
            return reward + self.discount_factor * next_q

    def _collect_human_feedback(self, disp):
        """Collect human feedback with timeout"""
        if not disp:
            return 0
            
        start_time = time.time()
        while time.time() - start_time < self.ts_len:
            reward = disp.get_feedback()
            if reward != 0:
                return reward
            time.sleep(0.01)
        return 0

    async def train(self, model_file_to_save=None):
        """Enhanced training loop"""
        print(f"Starting enhanced TAMER training - Mode: {self.mode}")
        
        # Question 3: Initialize appropriate interface
        from tamer.interface_enhanced import EnhancedInterface
        action_map = self._get_action_map()
        disp = EnhancedInterface(action_map, self.feedback_interface_type)

        for i in range(self.num_episodes):
            self._train_episode(i, disp)

        print('\nTraining completed!')
        self.env.close()
        
        if model_file_to_save:
            self.save_model(model_file_to_save)

    def _get_action_map(self):
        """Get action mapping for current environment"""
        if 'MountainCar' in str(self.env):
            return {0: '← left', 1: '○ none', 2: '→ right'}
        elif 'CartPole' in str(self.env):
            return {0: '← left', 1: '→ right'}
        else:
            return {i: str(i) for i in range(self.env.action_space.n)}

    def evaluate(self, n_episodes=10):
        """Enhanced evaluation with analysis"""
        print(f"\nEvaluating agent over {n_episodes} episodes...")
        rewards = []
        
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                
            rewards.append(total_reward)
            print(f'Episode {i+1}: Reward = {total_reward}')

        avg_reward = np.mean(rewards)
        print(f'Average reward: {avg_reward:.2f}')
        return avg_reward

    def analyze_performance(self):
        """Analyze and display performance differences between modes"""
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS")
        print("="*50)
        
        if hasattr(self, 'performance_history') and self.performance_history:
            final_reward = self.performance_history[-1]['total_reward']
            print(f"Final Episode Reward: {final_reward}")
            
        # Question 4: Highlight learning differences
        print(f"\nLearning Mode: {self.mode}")
        if self.mode == 'tamer':
            print("- Update Type: Supervised learning from human feedback")
            print("- No bootstrapping, direct reward assignment")
            print("- Dependent on human input quality and frequency")
        elif self.mode == 'q_learning':
            print("- Update Type: Temporal Difference learning")
            print("- Bootstrapping from future state estimates") 
            print("- Learns from environmental rewards only")
        elif self.mode == 'hybrid':
            print("- Update Type: Combined TD and supervised learning")
            print("- Benefits from both human guidance and environmental feedback")
            print("- More robust but requires careful tuning")

    def _save_analysis_data(self):
        """Save analysis data for reporting"""
        analysis = {
            'training_id': str(self.uuid),
            'mode': self.mode,
            'performance_history': self.performance_history,
            'hyperparameters': {
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'shaping_lambda': self.shaping_lambda,
                'transfer_alpha': self.transfer_alpha
            }
        }
        
        with open(self.stats_path, 'w') as f:
            json.dump(analysis, f, indent=2)

    def save_model(self, filename):
        """Enhanced model saving"""
        model_data = {
            'mode': self.mode,
            'performance_history': self.performance_history
        }
        
        if hasattr(self, 'H'):
            model_data['H'] = self.H
        if hasattr(self, 'Q'):
            model_data['Q'] = self.Q
            
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filename):
        """Enhanced model loading"""
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model_data = pickle.load(f)
            
        self.mode = model_data.get('mode', 'tamer')
        self.performance_history = model_data.get('performance_history', [])
        
        if 'H' in model_data:
            self.H = model_data['H']
        if 'Q' in model_data:
            self.Q = model_data['Q']