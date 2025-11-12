import numpy as np
from grid2op import Environment

class ReplayBuffer:

    def __init__(self, capacity=5000):
        transition_type_str = self.get_transition_type_str()
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(state_dim: int = 493, action_dim: int = 178) -> np.dtype:
        """
        Build a NumPy structured dtype for transitions:
        (state, action_vector, reward, next_state, done).

        - state:      float32[state_dim]
        - action:     float32[action_dim]   (vector form)
        - reward:     float32
        - next_state: float32[state_dim]
        - done:       bool
        """
        return np.dtype([
            ('state',      np.float32, (state_dim,)),
            ('action',     np.float32, (action_dim,)),
            ('reward',     np.float32),
            ('next_state', np.float32, (state_dim,)),
            ('done',       np.bool_)
        ])

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count