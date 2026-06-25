# Deep Learning Applications — Laboratory 3
## Deep Reinforcement Learning: REINFORCE and DQN

This lab is a hands-on study of deep reinforcement learning, covering policy gradient methods and value-based off-policy learning. The work is organized in three exercises: an improved implementation of REINFORCE with ablations on standardization and learning rate, an Actor-Critic extension with a learned value baseline, and a full DQN implementation tested on both CartPole and LunarLander.

All algorithms are implemented in a clean, importable package (`drl_lab`) with separate modules for networks, algorithms, replay buffer, and evaluation. Configurations are managed via OmegaConf YAML files; experiments are tracked with Weights & Biases.

---

## Environments

<table>
<tr>
<td align="center" width="50%">

### CartPole-v1

![CartPole environment](img/cart_pole.gif)

**Observation space:** 4 continuous dimensions (cart position, cart velocity, pole angle, pole angular velocity)  
**Action space:** Discrete(2) — push left or push right  
**Reward:** +1 for every timestep the pole stays upright  
**Solved at:** 500 (maximum episode length)  
**Random baseline:** < 20 steps on average

</td>
<td align="center" width="50%">

### LunarLander-v3

![LunarLander environment](img/lunar_lander.gif)

**Observation space:** 8 continuous dimensions (position, velocity, angle, angular velocity, leg contacts)  
**Action space:** Discrete(4) — do nothing, fire left engine, fire main engine, fire right engine  
**Reward:** shaped reward for soft landing between flags; penalty for crashing or using fuel  
**Solved at:** ~200 average reward  
**Random baseline:** strongly negative (frequent crashes)

</td>
</tr>
</table>

---

## Exercise 1 — REINFORCE with Standardization Ablations

### Algorithm

REINFORCE is an on-policy Monte Carlo policy gradient method. At each episode, a full trajectory is collected, discounted returns $G_t$ are computed backwards through time, and the policy is updated via:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \, G_t \, \nabla \log \pi(A_t \mid S_t, \boldsymbol{\theta})$$

The key improvement explored here is **return standardization**: before computing the policy gradient, the returns are normalized to zero mean and unit variance per episode. This keeps gradient magnitudes consistent regardless of the absolute scale of rewards in that episode, reducing variance across updates.

The evaluation methodology was also improved: every $N$ episodes, the agent is run for $M$ deterministic (greedy) episodes and the average total reward and average episode length are logged — a much more reliable signal than the noisy single-episode training reward.

### Network Architecture

**PolicyNetwork:** 2-layer MLP — `Linear(4, 128)` → ReLU → `Linear(128, 2)` → Softmax. Outputs a probability distribution over actions from which actions are sampled during training and argmax is taken during evaluation.

### Training Configuration

| Hyperparameter | Value |
|:---|:---:|
| Optimizer | Adam |
| Discount factor γ | 0.99 |
| Total episodes | 700 |
| Evaluation frequency N | 50 episodes |
| Evaluation episodes M | 10 |
| Seed | 211 |
| Device | CPU |

### Results — REINFORCE Ablations on CartPole

All runs start from the same seed (211) and same initial policy weights.

| Configuration | Best eval reward | Convergence episode | Stability |
|:---|:---:|:---:|:---:|
| No standardize, lr=1e-2 | ~151 (ep 50) | Never stable | Collapses after ep 100 |
| No standardize, lr=3e-3 | ~337 (ep 150) | Unstable | Oscillates, never consistent |
| **Standardize, lr=1e-2** | **500** | **ep 250** | Good, two dips at ep 200/450 |
| **Standardize, lr=3e-3** | **500** | **ep 150** | Best — smaller dips, more robust |

![REINFORCE ablation curves](doc/reinforce_ablations.png)
*Evaluation reward curves for the four REINFORCE configurations. Green dashed line = solved threshold (500).*

**Key finding:** standardization is the decisive factor. Without it, both learning rates lead to instability — the higher lr collapses fast (peaks at ~151 at ep 50 then diverges), the lower lr recovers partially (peak ~337 at ep 150) but never converges reliably. With standardization, both lr values solve CartPole, with lr=3e-3 being marginally more robust (smaller reward dip at ep 600 compared to lr=1e-2's dips at ep 200 and 450). The standardization=True, lr=1e-2 configuration had been used as the baseline for Exercise 1 before the ablations.

---

## Exercise 2 — REINFORCE with Value Baseline (Actor-Critic)

### Algorithm Extension

The value baseline extends REINFORCE by subtracting a learned state-value estimate $\tilde{V}(s)$ from the return:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \underbrace{(G_t - \tilde{V}(S_t))}_{\text{advantage}} \nabla \log \pi(A_t \mid S_t, \boldsymbol{\theta})$$

The advantage $G_t - \tilde{V}(S_t)$ captures how much better (or worse) the actual return was compared to what the critic expected for that state. This is **state-dependent variance reduction**: unlike standardization which normalizes globally per episode, the value baseline reduces variance in a semantically meaningful way — updates are large when the agent behaves surprisingly well or badly, and small when the return matches expectations.

The critic is trained simultaneously via MSE loss against the Monte Carlo return, then **detached** before computing the actor gradient. The `.detach()` is essential: without it, the actor's backward pass would flow gradients through the critic's computation graph, corrupting both updates.

### Additional Network

**ValueNetwork:** 2-layer MLP — `Linear(4, 128)` → ReLU → `Linear(128, 1)`. Single scalar output with no final activation, since $V(s)$ is unbounded.

### Training Configuration

| Hyperparameter | Value |
|:---|:---:|
| Actor optimizer | Adam, lr=1e-2 |
| Critic optimizer | Adam, lr=1e-2 |
| Standardization | Disabled (baseline handles variance) |
| All other params | Same as Exercise 1 |

### Results

| Configuration | Best eval reward | Convergence episode |
|:---|:---:|:---:|
| Standardize only, lr=1e-2 | 500 | ep 250 |
| **Value baseline, lr=1e-2** | **500** | **ep 150–200** |

The value baseline converges to maximum reward roughly 50–100 episodes earlier than the best standardization-only configuration. From ep 200 onwards the agent holds 500 consistently with no significant collapses, whereas the standardization runs show instability at various points. This confirms the theoretical advantage of a state-dependent baseline over a global normalization scheme.

---

## Exercise 3.2 — Deep Q-Network (DQN)

The chosen exercise is **Exercise 3.2: Solving CartPole and LunarLander with Deep Q-Learning**.

### Algorithm

DQN is an off-policy, step-based value learning algorithm. Instead of collecting full episodes like REINFORCE, DQN interacts with the environment one step at a time, stores transitions $(s, a, r, s', \text{done})$ in a **replay buffer**, and samples random mini-batches to train a Q-network to predict action-values. The TD target is computed using a separate **target network** that is updated slowly, decoupling the moving target problem that destabilizes naive Q-learning.

Two key design choices were made over the standard DQN:
- **Soft target updates** (τ=0.005) instead of periodic hard copies: `θ_target ← τ·θ_online + (1-τ)·θ_target`. This provides smoother, more stable target evolution.
- **Linear ε-greedy decay**: ε goes from 1.0 to 0.05 over the first `epsilon_decay` steps, then stays fixed. Early steps are purely exploratory; later steps are mostly greedy.

### Network Architecture

**QNetwork:** 3-layer MLP — `Linear(state_dim, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, action_dim)`. Outputs a Q-value for each action simultaneously; the agent takes `argmax` over these values.

### Training Configuration

| Hyperparameter | CartPole | LunarLander |
|:---|:---:|:---:|
| Optimizer | AdamW, lr=1e-3 | AdamW, lr=1e-3 |
| Total steps | ~107k (interrupted) | 200k |
| Replay buffer capacity | (from config) | (from config) |
| Batch size | (from config) | (from config) |
| Learning starts | (from config) | (from config) |
| ε start → end | 1.0 → 0.05 | 1.0 → 0.05 |
| ε decay steps | (from config) | (from config) |
| Soft update τ | 0.005 | 0.005 |
| Loss | MSE | Huber (smooth L1) |
| Gradient clip | 100 | 100 |
| γ | 0.99 | 0.99 |

### Results — CartPole

Training was interrupted at ~107k steps. The DQN shows the typical learning profile: step 1000 (high ε, purely random) eval reward of ~9–13, rapid improvement at step 2000 (eval ~167–229), then oscillation as ε decays and the buffer fills. The soft target update keeps learning stable without the oscillations typical of hard update DQN.

![DQN CartPole curves](doc/dqn_cartpole.png)
*DQN evaluation reward and episode length on CartPole. Training stopped at ~107k steps.*

### Results — LunarLander

Full 200k step training. LunarLander is significantly harder: at step 1000 (high ε) eval reward is -296.0 with episode length 72.4 (frequent early crashes). By step 2000 the agent already learns to avoid immediate crashes (reward -112.7, length 1000 — episodes now run to timeout rather than crashing). Progress is slower than CartPole throughout training due to the denser action space (4 actions), more complex dynamics, and shaped reward that requires learning to coordinate the engines.

The Huber loss (instead of MSE used for CartPole) was chosen for LunarLander because it is less sensitive to large TD errors early in training when Q-value estimates are far from accurate — this helps stabilize the early phase where the agent is essentially random.

![DQN LunarLander curves](doc/dqn_lunarlander.png)
*DQN evaluation reward and episode length on LunarLander-v3 over 200k steps.*

---

## Code Structure

```
drl_lab/
├── networks.py      # PolicyNetwork, ValueNetwork, QNetwork
├── algorithms.py    # reinforce(), train_dqn(), run_episode(), select_action()
├── buffer.py        # ReplayBuffer (deque-based, random sampling)
├── evaluation.py    # evaluate_policy() — greedy deterministic evaluation
config_base.yaml     # REINFORCE hyperparameters
config_dqn.yaml      # DQN hyperparameters
```

---

## Notes

- The `.detach()` on advantages in the Actor-Critic update is not optional — without it, `actor_loss.backward()` would compute gradients through the critic's computation graph, corrupting the critic's parameters on the actor's optimizer step.
- `evaluate_policy()` uses `argmax` (greedy) action selection regardless of whether the model is a policy network (which outputs probabilities) or a Q-network (which outputs Q-values). This works correctly in both cases since `argmax(softmax(x)) = argmax(x)`.
- The replay buffer uses `deque(maxlen=capacity)` which automatically evicts the oldest transitions when full — no explicit management needed.
- Soft target updates with τ=0.005 mean the target network moves very slowly: after 1000 gradient steps it has absorbed only ~1-(1-0.005)^1000 ≈ 99.3% of the online network — but the process is continuous and smooth rather than a sudden jump every N steps.
- The `UserWarning: step() called after terminated=True` appears once per episode end due to a minor loop ordering issue — one extra transition per episode is pushed to the buffer but has negligible effect given buffer sizes in the thousands.
