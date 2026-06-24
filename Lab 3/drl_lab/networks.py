import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    A simple feedforward policy network for discrete action spaces.
    Outputs a probability distribution over actions via softmax.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action probabilities."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
    

class ValueNetwork(nn.Module):
    """
    A simple feedforward value network (Critic) for state-value estimation.
    Outputs a single scalar representing V(s).
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Single output node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw state value."""
        x = F.relu(self.fc1(x))
        return self.fc2(x) # No activation on the output
    
    
    
    
class QNetwork(nn.Module):
    """
    Q-Network for estimating Q-values in discrete action spaces.
    Outputs a vector of Q-values for each action given a state.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for each action."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 