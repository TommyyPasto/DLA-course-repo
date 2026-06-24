import random
import numpy as np
import torch
from collections import deque
from typing import Tuple

class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy algorithms like DQN.
    Memorizes transitions (state, action, reward, next_state, done) and allows random sampling for training.
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Adds a transition to the buffer.
        
        Args:
            state: Current state (observation).
            action: Action taken.
            reward: Reward received after taking the action.    
            next_state: Next state (observation) after taking the action.
            done: Boolean indicating if the episode terminated after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as torch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)).unsqueeze(1),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1)
        )

    def __len__(self) -> int:
        return len(self.buffer)