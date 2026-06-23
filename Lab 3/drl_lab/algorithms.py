import numpy as np
import torch
from torch.distributions import Categorical
from .evaluation import evaluate_policy
import os
import torch.nn.functional as F


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



def reinforce(policy, env, value_net=None, optimizer=None, value_optimizer=None, lr=1e-2, value_lr=1e-2,
              gamma=0.99, num_episodes=1000, eval_freq=100, eval_episodes=10, 
              device="cpu", use_standardize=True, use_wandb=False, checkpoint_dir=None):
    """
    REINFORCE algorithm with optional Standardize Baseline and optional Value Baseline.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr)
        
    # Setup Value Optimizer if a value_net is provided
    if value_net is not None and value_optimizer is None:
        print("⚠️  Value network provided but no optimizer specified. Defaulting to Adam.")
        value_optimizer = torch.optim.Adam(value_net.parameters(), value_lr)

    best_eval_reward = -float('inf')
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history = {"train_rewards": [], "eval_episodes": [], "eval_rewards": [], "eval_lengths": []}

    policy.train()
    if value_net is not None: value_net.train()

    for episode in range(num_episodes):
        # 1. Collect Trajectory
        observations, _, log_probs, rewards = run_episode(env, policy, device=device)
        
        # Stack observations from list of (1, state_dim) to (T, state_dim)
        states = torch.cat(observations) 
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)
        
        # 2. Compute Target (Advantage or Standardized Returns)
        if value_net is not None:
            # --- VALUE BASELINE ---
            values = value_net(states).squeeze(-1) # Shape: (T,)
            
            # 1. Aggiorna il Critic (Value Network)
            value_optimizer.zero_grad()
            critic_loss = F.mse_loss(values, returns)
            critic_loss.backward() # Qui il grafo del Critic viene distrutto (ed è giusto così!)
            value_optimizer.step()
            
            # 2. Calcola l'Advantage STACCANDOLO dal grafo del Critic
            # .detach() dice a PyTorch: "usa solo il numero, non propagare gradienti al Critic"
            advantages = returns - values.detach() # <--- IL FIX MAGICO
            
            target = advantages
        else:
            # --- STANDARD REINFORCE ---
            target = returns
            critic_loss = torch.tensor(0.0) # Dummy for logging

        # Optional: Standardize the target (First Things First requirement)
        if use_standardize:
            target = (target - target.mean()) / (target.std() + 1e-8)
            
        # 3. Update Actor (Policy Network)
        optimizer.zero_grad()
        actor_loss = (-log_probs * target).mean()
        actor_loss.backward()
        optimizer.step()
        
        episode_reward = sum(rewards)
        history["train_rewards"].append(episode_reward)
        
        if use_wandb:
            import wandb
            log_dict = {
                "train/episode_reward": episode_reward, 
                "train/actor_loss": actor_loss.item()
            }
            if value_net is not None:
                log_dict["train/critic_loss"] = critic_loss.item()
            wandb.log(log_dict, step=episode)

        # 4. Periodic Evaluation
        if episode % eval_freq == 0:
            avg_reward, avg_length = evaluate_policy(env, policy, eval_episodes, device)
            history["eval_episodes"].append(episode)
            history["eval_rewards"].append(avg_reward)
            history["eval_lengths"].append(avg_length)
            
            if use_wandb:
                import wandb
                wandb.log({"eval/avg_reward": avg_reward, "eval/avg_length": avg_length}, step=episode)
            
            print(f"[Ep {episode:>4d}] Train: {episode_reward:6.1f} | Eval: {avg_reward:6.1f}")
            
            if checkpoint_dir is not None and avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                save_path = os.path.join(checkpoint_dir, "best_policy.pth")
                torch.save({
                    'episode': episode,
                    'actor_state_dict': policy.state_dict(),
                    'critic_state_dict': value_net.state_dict() if value_net else None,
                    'reward': avg_reward,
                }, save_path)
                if use_wandb: wandb.save(save_path)
                print(f"   💾 New best model saved!")
    if use_wandb:
        wandb.finish()
                
    return history