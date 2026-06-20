import torch
import numpy as np


def evaluate_policy(env, policy: torch.nn.Module, num_episodes: int = 10, device: str = "cpu") -> tuple[float, float]:
    """
    Evaluate policy performance over M episodes without updating weights.
    Uses greedy action selection for deterministic assessment.
    
    Args:
        env: Gymnasium environment.
        policy: Trained policy network.
        num_episodes: Number of evaluation episodes (M).
        device: Compute device.
        
    Returns:
        Tuple of (average_total_reward, average_episode_length).
    """
    policy.eval()
    total_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            done = False
            
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                probs = policy(obs_tensor)
                # Greedy selection for evaluation
                action = torch.argmax(probs, dim=-1).item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated
                
            total_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            
    policy.train()  # Always restore training mode after evaluation
    return float(np.mean(total_rewards)), float(np.mean(episode_lengths))