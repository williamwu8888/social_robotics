import numpy as np

# GridWorldEnvironment class
class GridWorldEnvironment:
    def __init__(self, grid_size=(5, 5), obstacles=[], stochastic=False):
        self.grid_size = grid_size
        self.state_space = np.prod(grid_size)
        self.action_space = 4  # Four possible actions: Up, Down, Left, Right
        self.state = None
        self.obstacles = obstacles
        self.set_mdp()

    def set_mdp(self):
        """ DEFINE THE MDP HERE"""
        n_states = self.state_space
        self.mdp = np.zeros((self.action_space, n_states, n_states))
        
        # Helper function to convert state to coordinates
        def to_coord(s):
            rows, cols = self.grid_size
            return divmod(s, cols)
        
        # Helper function to convert coordinates to state
        def to_state(r, c):
            rows, cols = self.grid_size
            return r * cols + c
        
        # Helper function to check if coordinates are valid and not obstacle
        def is_valid(r, c):
            s = to_state(r, c)
            return (0 <= r < self.grid_size[0] and 
                    0 <= c < self.grid_size[1] and 
                    s not in self.obstacles)
        
        # Build transition matrices for each action
        for s in range(n_states):
            r, c = to_coord(s)
            
            for a in range(self.action_space):
                nr, nc = r, c  # new row, new column
                
                if a == 0:     # UP
                    nr = max(0, r - 1)
                elif a == 1:   # DOWN
                    nr = min(self.grid_size[0] - 1, r + 1)
                elif a == 2:   # LEFT
                    nc = max(0, c - 1)
                elif a == 3:   # RIGHT
                    nc = min(self.grid_size[1] - 1, c + 1)
                
                # Check if new position is valid
                if is_valid(nr, nc):
                    ns = to_state(nr, nc)
                else:
                    ns = s  # Stay in current state if invalid move
                
                # Set deterministic transition probability
                self.mdp[a, s, ns] = 1.0

    def reset(self):
        # Randomly select an initial state that is not an obstacle
        valid_states = [s for s in range(self.state_space) 
                    if s not in self.obstacles]
        random_state = np.random.choice(valid_states)
        self.state = int(random_state)  # Convert to regular Python int
        return int(random_state)  # Convert to regular Python int

    def step(self, action):
        if self.state is None:
            raise Exception("You must call reset() before step()")

        # Use the MDP to get transition probabilities for current state and action
        transition_probs = self.mdp[action, self.state]
        
        # Sample next state from the probability distribution
        next_state = np.random.choice(self.state_space, p=transition_probs)

        # Define the reward function (e.g., reaching a goal state)
        done = (next_state == self.state_space - 1)  # Goal state
        reward = 1 if done else 0

        # Update the state
        self.state = next_state

        return self.state, reward, done

    def print(self):
        """ IMPLEMENT THE PRINT METHOD HERE """
        rows, cols = self.grid_size
        
        # Helper function to convert state to coordinates
        def to_coord(s):
            return divmod(s, cols)
        
        # Create grid representation
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                state = r * cols + c
                
                if state in self.obstacles:
                    row.append('X')  # Obstacle
                elif state == self.state_space - 1:
                    row.append('G')  # Goal
                elif self.state is not None and state == self.state:
                    row.append('A')  # Agent
                else:
                    row.append('.')  # Empty cell
            grid.append(row)
        
        # Print the grid
        for row in grid:
            print(' '.join(row))
        print()


class IcyGridWorldEnvironment(GridWorldEnvironment):
    def __init__(self, grid_size=(5, 5), obstacles=[]):
        # Call parent class constructor properly
        super().__init__(grid_size, obstacles)
        self.set_mdp()  # This will override the parent's MDP with icy version

    def set_mdp(self):
        """ DEFINE THE ICY-STOCHASTIC MDP HERE """
        n_states = self.state_space
        self.mdp = np.zeros((self.action_space, n_states, n_states))
        
        # Helper functions
        def to_coord(s):
            rows, cols = self.grid_size
            return divmod(s, cols)
        
        def to_state(r, c):
            rows, cols = self.grid_size
            return r * cols + c
        
        def is_valid(r, c):
            s = to_state(r, c)
            return (0 <= r < self.grid_size[0] and 
                    0 <= c < self.grid_size[1] and 
                    s not in self.obstacles)
        
        # Build stochastic transition matrices for icy world with inertia
        for s in range(n_states):
            r, c = to_coord(s)
            
            for a in range(self.action_space):
                # Calculate intended direction
                if a == 0:     # UP
                    direction = (-1, 0)
                elif a == 1:   # DOWN
                    direction = (1, 0)
                elif a == 2:   # LEFT
                    direction = (0, -1)
                elif a == 3:   # RIGHT
                    direction = (0, 1)
                
                # Calculate all reachable states in this direction with power law probabilities
                reachable_states = []
                probabilities = []
                
                # Start with current state (possibility of no movement)
                reachable_states.append(s)
                probabilities.append(0.1)  # Base probability for no movement
                
                # Explore states along the action direction with decreasing probabilities
                max_slide = 3  # Maximum slide distance
                dr, dc = direction
                
                for slide_distance in range(1, max_slide + 1):
                    new_r = r + dr * slide_distance
                    new_c = c + dc * slide_distance
                    
                    # Bound checking
                    new_r = max(0, min(self.grid_size[0] - 1, new_r))
                    new_c = max(0, min(self.grid_size[1] - 1, new_c))
                    
                    new_state = to_state(new_r, new_c)
                    
                    # If we hit an obstacle, stop sliding and don't add this state
                    if not is_valid(new_r, new_c):
                        break
                    
                    # Power law: probability decreases with distance
                    prob = 0.9 * (0.3 ** slide_distance)  # Adjust these values as needed
                    reachable_states.append(new_state)
                    probabilities.append(prob)
                
                # Normalize probabilities to sum to 1
                total_prob = sum(probabilities)
                if abs(total_prob - 1.0) > 1e-10:  # Only normalize if not already 1
                    probabilities = [p / total_prob for p in probabilities]
                
                # Assign probabilities to the MDP
                for state_idx, prob in zip(reachable_states, probabilities):
                    self.mdp[a, s, state_idx] = prob
                
                # Verify probabilities sum to 1 (with small tolerance)
                prob_sum = np.sum(self.mdp[a, s])
                if abs(prob_sum - 1.0) > 1e-10:
                    # If still not summing to 1, force normalization
                    self.mdp[a, s] = self.mdp[a, s] / prob_sum
                    print(f"Fixed probabilities for state {s}, action {a}: was {prob_sum}, now {np.sum(self.mdp[a, s])}")