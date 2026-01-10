# Step 1: Standard library imports
import argparse
import collections
import os
import random
import time
# Step 2: Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Step 3: Hyperparameter defaults (can be overridden via CLI)
DEFAULT_GAMMA = 0.99
DEFAULT_ALPHA = 0.2
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR_ACTOR = 3e-4
DEFAULT_LR_CRITIC = 3e-4
DEFAULT_LR_ALPHA = 3e-4
DEFAULT_REPLAY_SIZE = 1_000_000
DEFAULT_WARMUP_EPISODES = 10
DEFAULT_MAX_EPISODES = 100
DEFAULT_MAX_STEPS_PER_EPISODE = 200
DEFAULT_EVAL_INTERVAL = 10
DEFAULT_UPDATES_PER_STEP = 1
LOG_STD_MIN = -20
LOG_STD_MAX = 2


# Step 4: Simple replay buffer to collect experience tuples
class ReplayBuffer:
    """Fixed-size buffer that stores transition tuples for off-policy updates."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones


# Step 5: Utility to softly copy parameters between networks
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-averages the parameters of the target network towards the source."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# Step 6: Actor network that outputs a squashed Gaussian policy
class PolicyNetwork(nn.Module):
    """Gaussian policy network with tanh squashing for bounded action spaces."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, action_scale: np.ndarray):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        # Store action scaling parameters as buffers (not trainable)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, state: torch.Tensor):
        """Returns mean and log_std for the Gaussian distribution."""
        hidden = self.net(state)
        mu = self.mu_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """Samples an action using the reparameterization trick.
        
        Returns:
            action: Scaled action for the environment
            log_prob: Log probability of the action
            mean_action: Deterministic action (for evaluation)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (tanh)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Deterministic action for evaluation
        mean = torch.tanh(mu) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


# Step 7: Q-network (critic) for value estimation
class QNetwork(nn.Module):
    """Q-network that estimates action-value function."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Returns Q(s, a)."""
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# Step 8: SAC Agent class that encapsulates all training logic
class SACAgent:
    """Soft Actor-Critic agent."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        device: torch.device,
        gamma: float = 0.99,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        hidden_size: int = 256,
        tau: float = 5e-3,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        
        # Initialize networks
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_size, action_scale).to(device)
        
        self.q1_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q2_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        
        self.q1_target_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q2_target_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())
        
        # Freeze target networks
        for param in self.q1_target_net.parameters():
            param.requires_grad = False
        for param in self.q2_target_net.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=lr_critic)
        
        # Entropy temperature
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, 
                                         device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device)
            self.alpha_optimizer = None
            self.target_entropy = None
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from the policy."""
        state_v = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, action_v = self.policy_net.sample(state_v)
        else:
            action_v, _, _ = self.policy_net.sample(state_v)
        return action_v.squeeze(0).cpu().numpy()
    
    def update(self, batch):
        """Perform one step of gradient descent on the networks."""
        states, actions, rewards, next_states, dones = batch
        
        states_v = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_v = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        alpha = self.log_alpha.exp().detach()
        
        # ============================================================
        # Update Q-networks (critics)
        # ============================================================
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions_v, next_log_probs_v, _ = self.policy_net.sample(next_states_v)
            
            # Compute target Q-values using target networks
            next_q1_v = self.q1_target_net(next_states_v, next_actions_v)
            next_q2_v = self.q2_target_net(next_states_v, next_actions_v)
            next_q_v = torch.min(next_q1_v, next_q2_v)
            
            # Target with entropy regularization
            target_q_v = rewards_v + self.gamma * (1.0 - dones_v) * (
                next_q_v - alpha * next_log_probs_v
            )
        
        # Current Q estimates
        q1_v = self.q1_net(states_v, actions_v)
        q2_v = self.q2_net(states_v, actions_v)
        
        # Q-network losses (MSE)
        q1_loss = F.mse_loss(q1_v, target_q_v)
        q2_loss = F.mse_loss(q2_v, target_q_v)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ============================================================
        # Update policy network (actor)
        # ============================================================
        # Sample new actions from current policy
        new_actions_v, log_probs_v, _ = self.policy_net.sample(states_v)
        
        # Compute Q-values for new actions
        q1_new_v = self.q1_net(states_v, new_actions_v)
        q2_new_v = self.q2_net(states_v, new_actions_v)
        min_q_new_v = torch.min(q1_new_v, q2_new_v)
        
        # Policy loss: maximize Q - alpha * log_prob
        policy_loss = (alpha * log_probs_v - min_q_new_v).mean()
        
        # Update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # ============================================================
        # Update entropy temperature (alpha)
        # ============================================================
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs_v.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # ============================================================
        # Soft update of target networks
        # ============================================================
        soft_update(self.q1_target_net, self.q1_net, self.tau)
        soft_update(self.q2_target_net, self.q2_net, self.tau)
        
        # Return losses for logging
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'entropy': -log_probs_v.mean().item(),
        }
    
    def save(self, path: str):
        """Save agent's networks."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'q1_state_dict': self.q1_net.state_dict(),
            'q2_state_dict': self.q2_net.state_dict(),
            'q1_target_state_dict': self.q1_target_net.state_dict(),
            'q2_target_state_dict': self.q2_target_net.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
    
    def load(self, path: str):
        """Load agent's networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.q1_net.load_state_dict(checkpoint['q1_state_dict'])
        self.q2_net.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target_net.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target_net.load_state_dict(checkpoint['q2_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha']


# Step 9: Evaluation helper
def evaluate_policy(env: gym.Env, agent: SACAgent, episodes: int = 10) -> tuple:
    """Evaluates the policy using deterministic actions."""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths)


# Step 10: Training function for one episode
def train_episode(
    env: gym.Env,
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    max_steps: int,
    batch_size: int,
    updates_per_step: int,
    warmup: bool = False,
) -> dict:
    """Execute one episode and perform training updates.
    
    Returns:
        Dictionary with episode statistics
    """
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    
    # Statistics for this episode
    total_q1_loss = 0.0
    total_q2_loss = 0.0
    total_policy_loss = 0.0
    total_alpha_loss = 0.0
    total_alpha = 0.0
    total_entropy = 0.0
    update_count = 0
    
    while not done and episode_length < max_steps:
        # Select action
        if warmup:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)
        
        # Execute action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        replay_buffer.push(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        episode_length += 1
        obs = next_obs
        
        # Perform updates (if not in warmup and enough samples)
        if not warmup and len(replay_buffer) >= batch_size:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                losses = agent.update(batch)
                
                total_q1_loss += losses['q1_loss']
                total_q2_loss += losses['q2_loss']
                total_policy_loss += losses['policy_loss']
                total_alpha_loss += losses['alpha_loss']
                total_alpha += losses['alpha']
                total_entropy += losses['entropy']
                update_count += 1
    
    # Compute average losses
    stats = {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
    }
    
    if update_count > 0:
        stats.update({
            'q1_loss': total_q1_loss / update_count,
            'q2_loss': total_q2_loss / update_count,
            'policy_loss': total_policy_loss / update_count,
            'alpha_loss': total_alpha_loss / update_count,
            'alpha': total_alpha / update_count,
            'entropy': total_entropy / update_count,
        })
    
    return stats


# Step 11: Main training routine
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAC agent on Pendulum-v1 (episode-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, 
                       help="Discount factor")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, 
                       help="Initial entropy coefficient")
    parser.add_argument("--auto-alpha", action="store_true", default=True,
                       help="Automatically tune entropy coefficient")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                       help="Batch size for updates")
    parser.add_argument("--lr-actor", type=float, default=DEFAULT_LR_ACTOR, 
                       help="Learning rate for the policy")
    parser.add_argument("--lr-critic", type=float, default=DEFAULT_LR_CRITIC, 
                       help="Learning rate for Q-networks")
    parser.add_argument("--lr-alpha", type=float, default=DEFAULT_LR_ALPHA, 
                       help="Learning rate for entropy temperature")
    parser.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE, 
                       help="Replay buffer capacity")
    parser.add_argument("--warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES, 
                       help="Random exploration episodes before training")
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES, 
                       help="Total number of training episodes")
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE,
                       help="Maximum steps per episode")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL, 
                       help="Episodes between policy evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--updates-per-step", type=int, default=DEFAULT_UPDATES_PER_STEP, 
                       help="Gradient updates per environment step")
    parser.add_argument("--hidden-size", type=int, default=256, 
                       help="Width of hidden layers")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs_sac_pendulum", 
                       help="TensorBoard log directory")
    parser.add_argument("--tau", type=float, default=5e-3, 
                       help="Target network smoothing coefficient")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Torch device (cpu, cuda, mps, auto)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Create environments
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 1)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = env.action_space.high
    
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Action scale: {action_scale}")

    # Initialize agent
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_scale=action_scale,
        device=device,
        gamma=args.gamma,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        hidden_size=args.hidden_size,
        tau=args.tau,
    )
    
    # Initialize replay buffer and logging
    replay_buffer = ReplayBuffer(args.replay_size)
    writer = SummaryWriter(log_dir=args.logdir)
    best_eval_reward = -float('inf')
    
    print(f"\nTraining for {args.max_episodes} episodes")
    print(f"Warmup: {args.warmup_episodes} episodes")
    print(f"Max steps per episode: {args.max_steps_per_episode}\n")
    
    start_time = time.time()
    total_steps = 0
    
    # ============================================================
    # Main training loop (episode-based)
    # ============================================================
    for episode in range(1, args.max_episodes + 1):
        # Determine if this is a warmup episode
        warmup = episode <= args.warmup_episodes
        
        # Train for one episode
        stats = train_episode(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            max_steps=args.max_steps_per_episode,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            warmup=warmup,
        )
        
        total_steps += stats['episode_length']
        
        # Log training statistics
        writer.add_scalar("train/episode_reward", stats['episode_reward'], episode)
        writer.add_scalar("train/episode_length", stats['episode_length'], episode)
        writer.add_scalar("train/total_steps", total_steps, episode)
        
        if not warmup:
            writer.add_scalar("loss/q1", stats['q1_loss'], episode)
            writer.add_scalar("loss/q2", stats['q2_loss'], episode)
            writer.add_scalar("loss/policy", stats['policy_loss'], episode)
            writer.add_scalar("loss/alpha", stats['alpha_loss'], episode)
            writer.add_scalar("alpha/value", stats['alpha'], episode)
            writer.add_scalar("policy/entropy", stats['entropy'], episode)
        
        # Print progress
        if episode % 10 == 0:
            if warmup:
                print(f"Episode {episode:4d} [WARMUP] | "
                      f"Reward: {stats['episode_reward']:7.2f} | "
                      f"Length: {stats['episode_length']:3d} | "
                      f"Buffer: {len(replay_buffer):6d}")
            else:
                print(f"Episode {episode:4d} | "
                      f"Reward: {stats['episode_reward']:7.2f} | "
                      f"Length: {stats['episode_length']:3d} | "
                      f"Q_loss: {stats['q1_loss']:6.3f} | "
                      f"Alpha: {stats['alpha']:.3f}")
        
        # Periodic evaluation
        if episode % args.eval_interval == 0:
            eval_reward, eval_std, eval_length = evaluate_policy(
                eval_env, agent, episodes=args.eval_episodes
            )
            
            writer.add_scalar("eval/mean_reward", eval_reward, episode)
            writer.add_scalar("eval/std_reward", eval_std, episode)
            writer.add_scalar("eval/mean_length", eval_length, episode)
                        # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                os.makedirs("saves", exist_ok=True)
                save_path = os.path.join("saves", "sac_pendulum_best.pth")
                agent.save(save_path)
                print(f"✓ Saved new best model (reward: {eval_reward:.2f})\n")

            elapsed_min = (time.time() - start_time) / 60.0
            print(f"\n{'='*70}")
            print(f"Evaluation at episode {episode} (total steps: {total_steps})")
            print(f"Mean reward: {eval_reward:.2f} ± {eval_std:.2f}")
            print(f"Mean length: {eval_length:.1f}")
            print(f"Best reward: {best_eval_reward:.2f}")
            print(f"Elapsed time: {elapsed_min:.1f} minutes")
            print(f"{'='*70}\n")
            

    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Training completed!")
    final_reward, final_std, final_length = evaluate_policy(
        eval_env, agent, episodes=20
    )
    print(f"Final evaluation (20 episodes):")
    print(f"  Mean reward: {final_reward:.2f} ± {final_std:.2f}")
    print(f"  Mean length: {final_length:.1f}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {(time.time() - start_time) / 60.0:.1f} minutes")
    print(f"{'='*70}\n")
    
    # Save final model
    final_save_path = os.path.join("saves", "sac_pendulum_final.pth")
    agent.save(final_save_path)
    print(f"Saved final model to {final_save_path}")
    
    writer.add_scalar("eval/final_reward", final_reward, args.max_episodes)
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()