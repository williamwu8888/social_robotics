# [file name]: tamer/interface_enhanced.py
"""
Enhanced interface addressing Question 3
- Multiple feedback modalities
- Improved visual feedback
- Support for different environments
"""

import os
import pygame
import numpy as np

class EnhancedInterface:
    """
    Enhanced interface supporting multiple feedback methods from Question 3
    """
    
    def __init__(self, action_map, interface_type='enhanced_keyboard'):
        self.action_map = action_map
        self.interface_type = interface_type
        self.last_feedback = None
        self.feedback_history = []
        
        pygame.init()
        
        # Configure based on interface type
        if interface_type == 'enhanced_keyboard':
            self.screen = pygame.display.set_mode((300, 150))
            self._setup_enhanced_keyboard()
        elif interface_type == 'simple_keyboard':
            self.screen = pygame.display.set_mode((200, 100))
            self._setup_simple_keyboard()
            
        pygame.display.set_caption(f"TAMER Controller - {interface_type}")
        
        # Position window
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        
        # Fonts
        self.font_large = pygame.font.Font("freesansbold.ttf", 24)
        self.font_small = pygame.font.Font("freesansbold.ttf", 16)
        
        self._draw_initial_interface()

    def _setup_enhanced_keyboard(self):
        """Question 3: Enhanced keyboard with multiple feedback levels"""
        self.key_mapping = {
            pygame.K_w: 2.0,    # Strong positive
            pygame.K_e: 1.0,    # Medium positive
            pygame.K_q: 0.5,    # Weak positive
            pygame.K_a: -2.0,   # Strong negative
            pygame.K_d: -1.0,   # Medium negative  
            pygame.K_s: -0.5,   # Weak negative
            pygame.K_SPACE: 0.0 # Explicit neutral
        }
        self.feedback_colors = {
            2.0: (0, 200, 0),    # Dark green
            1.0: (0, 255, 0),    # Green
            0.5: (150, 255, 150),# Light green
            -0.5: (255, 150, 150),# Light red
            -1.0: (255, 0, 0),   # Red
            -2.0: (200, 0, 0),   # Dark red
            0.0: (0, 0, 200)     # Blue for neutral
        }

    def _setup_simple_keyboard(self):
        """Original simple keyboard interface"""
        self.key_mapping = {
            pygame.K_w: 1.0,    # Positive
            pygame.K_a: -1.0,   # Negative
        }
        self.feedback_colors = {
            1.0: (0, 255, 0),   # Green
            -1.0: (255, 0, 0),  # Red
        }

    def _draw_initial_interface(self):
        """Draw initial interface state"""
        self.screen.fill((40, 40, 40))
        
        # Title
        title = self.font_small.render("TAMER Controller", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        # Instructions based on interface type
        if self.interface_type == 'enhanced_keyboard':
            instructions = [
                "W: Strong Positive (+2)",
                "E: Positive (+1)", 
                "Q: Weak Positive (+0.5)",
                "A: Strong Negative (-2)",
                "D: Negative (-1)",
                "S: Weak Negative (-0.5)",
                "Space: Neutral (0)"
            ]
        else:
            instructions = [
                "W: Positive Feedback",
                "A: Negative Feedback",
            ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (10, 30 + i * 18))
        
        pygame.display.update()

    def get_feedback(self):
        """
        Get human feedback - main method addressing Question 3
        Returns: scalar reward value
        """
        reward = 0
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.key_mapping:
                    reward = self.key_mapping[event.key]
                    self.last_feedback = reward
                    self.feedback_history.append({
                        'timestamp': pygame.time.get_ticks(),
                        'value': reward
                    })
                    self._visualize_feedback(reward)
                    break
        
        return reward

    def _visualize_feedback(self, reward):
        """Enhanced visual feedback for different levels"""
        if reward in self.feedback_colors:
            self.screen.fill(self.feedback_colors[reward])
        else:
            self.screen.fill((100, 100, 100))  # Default gray
            
        # Display feedback value
        if reward > 0:
            feedback_text = f"+{reward}"
            color = (255, 255, 255)
        elif reward < 0:
            feedback_text = f"{reward}"
            color = (255, 255, 255)
        else:
            feedback_text = "Neutral"
            color = (255, 255, 255)
            
        text_surface = self.font_large.render(feedback_text, True, color)
        text_rect = text_surface.get_rect(center=(150, 50))
        self.screen.blit(text_surface, text_rect)
        
        # Feedback type label
        type_label = self.font_small.render("Human Feedback", True, (255, 255, 255))
        type_rect = type_label.get_rect(center=(150, 90))
        self.screen.blit(type_label, type_rect)
        
        pygame.display.update()

    def show_action(self, action):
        """
        Show agent's current action with enhanced visualization
        """
        self.screen.fill((40, 40, 40))  # Reset to dark background
        
        # Current action
        action_text = self.font_large.render(
            self.action_map.get(action, str(action)), 
            True, (255, 255, 0)  # Yellow for action
        )
        action_rect = action_text.get_rect(center=(150, 50))
        self.screen.blit(action_text, action_rect)
        
        # Label
        label = self.font_small.render("Agent Action", True, (200, 200, 200))
        label_rect = label.get_rect(center=(150, 90))
        self.screen.blit(label, label_rect)
        
        # Last feedback if available
        if self.last_feedback is not None:
            feedback_color = (0, 255, 0) if self.last_feedback > 0 else \
                           (255, 0, 0) if self.last_feedback < 0 else \
                           (0, 0, 255)
            
            feedback_str = f"+{self.last_feedback}" if self.last_feedback > 0 else str(self.last_feedback)
            feedback_text = self.font_small.render(f"Last: {feedback_str}", True, feedback_color)
            self.screen.blit(feedback_text, (10, 120))
        
        pygame.display.update()

    def get_feedback_statistics(self):
        """Get statistics about human feedback patterns"""
        if not self.feedback_history:
            return None
            
        values = [f['value'] for f in self.feedback_history]
        return {
            'total_feedbacks': len(values),
            'positive_count': sum(1 for v in values if v > 0),
            'negative_count': sum(1 for v in values if v < 0),
            'neutral_count': sum(1 for v in values if v == 0),
            'average_feedback': np.mean(values) if values else 0,
            'feedback_frequency': len(values) / (pygame.time.get_ticks() / 1000) if pygame.time.get_ticks() > 0 else 0
        }