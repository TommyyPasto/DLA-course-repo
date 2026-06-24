import numpy as np
import torch
from torch.distributions import Categorical
from .evaluation import evaluate_policy
import os
import torch.nn.functional as F


def select_action(policy: torch.nn.Module, obs: torch.Tensor) -> tuple[int, torch.Tensor]:
    """
    Sample an action from the policy distribution.
    
    Args:
        policy: The policy network (actor).
        obs: Current observation/state as a torch tensor.
    
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
    
    Args:
        env: Gymnasium environment.
        policy: Policy network (actor).
        maxlen: Maximum number of steps to run in the episode.
        device: Compute device (cpu or cuda).
    
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
    
    Args:
        policy: Policy network (actor).
        env: Gymnasium environment.
        value_net: Optional value network (critic) for baseline.
        optimizer: Optimizer for policy network.
        value_optimizer: Optimizer for value network.
        lr: Learning rate for policy optimizer.
        value_lr: Learning rate for value optimizer.
        gamma: Discount factor.
        num_episodes: Total training episodes.
        eval_freq: Frequency of evaluation episodes.
        eval_episodes: Number of episodes per evaluation run.
        device: Compute device (cpu or cuda).
        use_standardize: Whether to standardize returns/advantages.
        use_wandb: Whether to log metrics to Weights & Biases.
        checkpoint_dir: Directory to save best model checkpoints.
    
    Returns:
        History dictionary containing training and evaluation metrics.
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
            # value baseline: compute V(s) for each state in the trajectory
            values = value_net(states).squeeze(-1) # Shape: (T,)
            
            # Update Critic (Value Network)
            value_optimizer.zero_grad()
            critic_loss = F.mse_loss(values, returns)
            critic_loss.backward() # Qui il grafo del Critic viene distrutto (ed è giusto così!)
            value_optimizer.step()
            
            # Compute advantages for Actor update. .detach() is crucial to prevent backprop through the value network when updating the policy.
            advantages = returns - values.detach() # <--- IL FIX MAGICO
            
            target = advantages
            
        else:
            # standard REINFORCE without baseline
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









def train_dqn(q_net, target_q_net, env, optimizer, buffer,
              gamma=0.99, total_steps=50000, eval_freq=1000, eval_episodes=10,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000,
              learning_starts=1000, batch_size=64, 
              #target_update_freq=100,
              tau=0.005,
              device="cpu", use_wandb=False, checkpoint_dir=None):
    """
    DQN Training Loop (Off-Policy, Step-Based).
    
    Args:
        q_net: Main Q-Network to be trained.
        target_q_net: Frozen Target Q-Network for stable TD targets.
        env: Gymnasium environment.
        optimizer: Torch optimizer for q_net.
        buffer: ReplayBuffer instance.
        gamma: Discount factor.
        total_steps: Total training steps.
        eval_freq: Evaluate every N steps.
        eval_episodes: Episodes per evaluation run.
        epsilon_start/end/decay: Epsilon-greedy exploration schedule.
        learning_starts: Steps before starting gradient updates.
        batch_size: Batch size for replay sampling.
        target_update_freq: Frequency for hard target network updates.
        device: Compute device.
        use_wandb: Enable W&B logging.
        checkpoint_dir: Directory to save best model.
        
    Returns:
        Dictionary with training and evaluation history.
    """
    q_net.train()
    target_q_net.eval() # Target network is always in eval mode
    
    best_eval_reward = -float('inf')
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    history = {"train_rewards": [], "eval_episodes": [], "eval_rewards": [], "eval_lengths": []}
    
    obs, _ = env.reset()
    episode_reward = 0.0
    step = 0
    
    # Initialize epsilon
    epsilon = epsilon_start
    
    if use_wandb:
        import wandb

    for step in range(1, total_steps + 1):
        # 1. Epsilon-Greedy Action Selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_net(obs_tensor)
                action = q_values.argmax(dim=-1).item()
                
        # 2. Environment Step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 3. Store in Replay Buffer
        buffer.push(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        obs = next_obs if not done else None
        
        # 4. Training Step (Gradient Update)
        loss = 0.0
        
        if len(buffer) >= learning_starts and len(buffer) >= batch_size:
            state_b, action_b, reward_b, next_state_b, done_b = buffer.sample(batch_size)
            state_b, action_b, reward_b, next_state_b, done_b = (
                state_b.to(device), action_b.to(device), reward_b.to(device),
                next_state_b.to(device), done_b.to(device)
            )
            
            with torch.no_grad():
                # TD Target: r + γ * max_a' Q_target(s', a') * (1 - done)
                max_next_q = target_q_net(next_state_b).max(dim=1, keepdim=True)[0]
                target_q = reward_b + gamma * max_next_q * (1 - done_b)
                
            current_q = q_net(state_b).gather(1, action_b)
            loss = F.smooth_l1_loss(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            
            #gradient clipping..
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=100)
            optimizer.step()
            
            # 5. Target Network Hard Update
            #if step % target_update_freq == 0:
            #    target_q_net.load_state_dict(q_net.state_dict())
            
            
            #  5. Target Network Soft Update(with tau)
            for target_param, online_param in zip(target_q_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            
        # 6. Epsilon Decay (Linear)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * min(step / epsilon_decay, 1.0))
        
        # 7. Episode End Handling
        if done:
            history["train_rewards"].append(episode_reward)
            if use_wandb:
                wandb.log({"train/episode_reward": episode_reward, "train/epsilon": epsilon, "train/loss": loss}, step=step)
                
            obs, _ = env.reset()
            episode_reward = 0.0
            
        # 8. Periodic Evaluation
        if step % eval_freq == 0:
            # evaluate_policy works with Q-networks because it uses argmax!
            avg_reward, avg_length = evaluate_policy(env, q_net, eval_episodes, device)
            history["eval_episodes"].append(step)
            history["eval_rewards"].append(avg_reward)
            history["eval_lengths"].append(avg_length)
            
            if use_wandb:
                wandb.log({"eval/avg_reward": avg_reward, "eval/avg_length": avg_length}, step=step)
            print(f"[Step {step:6d}] Eval: {avg_reward:6.1f} | Length: {avg_length:.1f} | ε: {epsilon:.3f}")
            
            # Checkpointing
            if checkpoint_dir is not None and avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                save_path = os.path.join(checkpoint_dir, "best_q_net.pth")
                torch.save({
                    'step': step,
                    'q_net_state_dict': q_net.state_dict(),
                    'target_q_net_state_dict': target_q_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': avg_reward,
                }, save_path)
                print(f"   💾 New best model saved!")
                
    if use_wandb:
        wandb.finish()
        
    return history