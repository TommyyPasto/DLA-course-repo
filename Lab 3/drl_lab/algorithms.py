import numpy as np
import torch
from torch.distributions import Categorical
from .evaluation import evaluate_policy
import os


def select_action(policy: torch.nn.Module, obs: torch.Tensor) -> tuple[int, torch.Tensor]:
    """
    Sample an action from the policy distribution.
    
    Returns:
        action: The sampled integer action.
        log_prob: The log probability of the sampled action (for gradient computation).
    """
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.reshape(1)


def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """
    Compute discounted returns backwards through the episode.
    
    Args:
        rewards: List of scalar rewards received at each step.
        gamma: Discount factor.
        
    Returns:
        Numpy array of discounted returns in chronological order.
    """
    returns = []
    discounted_sum = 0.0
    for r in reversed(rewards):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return np.array(returns)


def run_episode(env, policy: torch.nn.Module, maxlen: int = 500, device: str = "cpu") -> tuple:
    """
    Run a single episode collecting trajectories.
    
    Returns:
        Tuple of (observations, actions, log_probs, rewards).
    """
    observations, actions, log_probs, rewards = [], [], [], []
    obs, _ = env.reset()
    
    for _ in range(maxlen):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action, log_prob = select_action(policy, obs_tensor)
        
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
            
    return observations, actions, torch.cat(log_probs), rewards


def reinforce_baseline(policy, env, optimizer=None, gamma=0.99, num_episodes=1000, 
              eval_freq=100, eval_episodes=10, device="cpu"):
    """
    REINFORCE algorithm with periodic evaluation.
    
    Args:
        policy: The policy network to train.
        env: Gymnasium environment.
        optimizer: Torch optimizer (defaults to Adam if None).
        gamma: Discount factor.
        num_episodes: Total training episodes.
        eval_freq: Evaluate every N episodes.
        eval_episodes: Number of episodes M to run during evaluation.
        device: Compute device ('cpu' or 'cuda').
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        
    # Import here to avoid circular dependencies if evaluation is in separate module
    from .evaluation import evaluate_policy
    
    history = {"train_rewards": [], "eval_rewards": [], "eval_lengths": []}
    
    policy.train()
    for episode in range(num_episodes):
        # Collect trajectory
        _, _, log_probs, rewards = run_episode(env, policy, device=device)
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)
        
        # Standardize returns with epsilon for numerical stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        optimizer.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        optimizer.step()
        
        history["train_rewards"].append(sum(rewards))
        
        # Periodic robust evaluation
        if episode % eval_freq == 0:
            avg_reward, avg_length = evaluate_policy(env, policy, eval_episodes, device)
            history["eval_rewards"].append(avg_reward)
            history["eval_lengths"].append(avg_length)
            print(f"[Ep {episode}] Eval Reward: {avg_reward:.2f} | Length: {avg_length:.1f}")
            
    return history


def reinforce(policy, env, optimizer=None, gamma=0.99, num_episodes=1000, 
              eval_freq=100, eval_episodes=10, device="cpu",
              use_wandb=False, checkpoint_dir=None):
    """
    REINFORCE algorithm with periodic robust evaluation.
    
    Args:
        policy: The policy network to train.
        env: Gymnasium environment.
        optimizer: Torch optimizer (defaults to Adam if None).
        gamma: Discount factor.
        num_episodes: Total training episodes.
        eval_freq: Run evaluation every N episodes.
        eval_episodes: Number of episodes M for each evaluation.
        device: Compute device ('cpu' or 'cuda').
        If use_wandb=True, logs metrics to Weights & Biases.
        If checkpoint_dir is provided, saves the best model based on eval reward.
        
    Returns:
        Dictionary with training and evaluation history.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    # Setup Checkpointing (only if requested)
    best_eval_reward = -float('inf')
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "train_rewards": [], "eval_episodes": [], 
        "eval_rewards": [], "eval_lengths": []
    }

    policy.train()
    for episode in range(num_episodes):
        # --- Training step ---
        _, _, log_probs, rewards = run_episode(env, policy, device=device)
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        optimizer.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        optimizer.step()
        
        episode_reward = sum(rewards)
        history["train_rewards"].append(episode_reward)
        
        # Log Training (only if requested)
        if use_wandb:
            import wandb
            wandb.log({"train/episode_reward": episode_reward, "train/loss": loss.item()}, step=episode)

        # --- Periodic evaluation ---
        if episode % eval_freq == 0:
            avg_reward, avg_length = evaluate_policy(env, policy, eval_episodes, device)
            history["eval_episodes"].append(episode)
            history["eval_rewards"].append(avg_reward)
            history["eval_lengths"].append(avg_length)
            
            # Log Eval (only if requested)
            if use_wandb:
                import wandb
                wandb.log({"eval/avg_reward": avg_reward, "eval/avg_length": avg_length}, step=episode)
            
            print(f"[Ep {episode:>4d}] Train: {episode_reward:6.1f} | Eval: {avg_reward:6.1f}")
            
            # Checkpointing (only if requested)
            if checkpoint_dir is not None and avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                save_path = os.path.join(checkpoint_dir, "best_policy.pth")
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': avg_reward,
                }, save_path)
                print(f"   💾 New best model saved!")
                
    return history
