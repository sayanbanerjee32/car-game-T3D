import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Critic Network (Q-value function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6, positive_sample_ratio=0.7, positive_sample_threshold=0):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.positive_sample_ratio = positive_sample_ratio
        self.positive_sample_threshold = positive_sample_threshold
        self.deletion_interval = 10000  # Delete every 10000 samples
        self.sample_count = 0

        # Initialize running statistics for rewards
        self.running_mean = None
        self.running_variance = None
        self.alpha = 0.99  # Smoothing factor for updating mean and variance
        self.update_interval = 1000  # Update stats every 1000 samples

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
        self.sample_count += 1

    def update_running_stats(self, rewards):
        if self.running_mean is None:
            self.running_mean = np.mean(rewards)
            self.running_variance = np.var(rewards)
        else:
            new_mean = np.mean(rewards)
            new_variance = np.var(rewards)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * new_mean
            self.running_variance = self.alpha * self.running_variance + (1 - self.alpha) * new_variance

    def normalize_reward(self, reward):
        if self.running_mean is None:
            return reward
        normalized = (reward - self.running_mean) / (np.sqrt(self.running_variance) + 1e-8)
        return np.clip(normalized, -5, 5)  # Clip to [-5, 5]

    def update_positive_threshold(self):
        if len(self.storage) > 1000:  # Only update after sufficient samples
            rewards = [t[-2] for t in self.storage]
            self.positive_sample_threshold = np.percentile(rewards, 95)  # Set threshold at 95th percentile

    def delete_negative_samples(self):
        pos_indices = [i for i, t in enumerate(self.storage) if t[-2] >= self.positive_sample_threshold]
        neg_indices = [i for i in range(len(self.storage)) if i not in pos_indices]
        
        initial_size = len(self.storage)
        
        # Calculate how many negative samples to keep
        n_neg_to_keep = max(int(len(pos_indices) / self.positive_sample_ratio) - len(pos_indices), 0)
        n_neg_to_keep = min(n_neg_to_keep, len(neg_indices))  # Ensure we don't keep more than available
        
        # Keep all positive samples and randomly select negative samples
        keep_neg = np.random.choice(neg_indices, size=n_neg_to_keep, replace=False)
        keep_indices = sorted(list(pos_indices) + list(keep_neg))
        
        self.storage = [self.storage[i] for i in keep_indices]
        self.ptr = len(self.storage) % self.max_size
        
        deleted_samples = initial_size - len(self.storage)
        print(f"Deleted {deleted_samples} negative samples. Remaining samples: {len(self.storage)}")

    def sample(self, batch_size):
        if self.sample_count >= self.deletion_interval:
            self.update_positive_threshold()
            self.delete_negative_samples()
            self.sample_count = 0

        total_samples = len(self.storage)
        if total_samples < batch_size:
            print(f"Warning: Not enough samples in buffer. Adjusting batch size from {batch_size} to {total_samples}")
            batch_size = total_samples

        pos_indices = [i for i, t in enumerate(self.storage) if t[-2] >= self.positive_sample_threshold]
        neg_indices = [i for i in range(total_samples) if i not in pos_indices]

        n_pos = min(int(batch_size * self.positive_sample_ratio), len(pos_indices))
        n_neg = batch_size - n_pos

        # Adjust n_pos and n_neg if there aren't enough samples of either type
        if len(pos_indices) < n_pos:
            n_pos = len(pos_indices)
            n_neg = min(batch_size - n_pos, len(neg_indices))
        elif len(neg_indices) < n_neg:
            n_neg = len(neg_indices)
            n_pos = min(batch_size - n_neg, len(pos_indices))

        # Sample with replacement if necessary
        pos_samples = np.random.choice(pos_indices, size=n_pos, replace=len(pos_indices) < n_pos)
        neg_samples = np.random.choice(neg_indices, size=n_neg, replace=len(neg_indices) < n_neg)

        ind = np.concatenate([pos_samples, neg_samples])
        np.random.shuffle(ind)

        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            
            # Update running stats and normalize reward
            self.update_running_stats([reward])
            normalized_reward = self.normalize_reward(reward)
            
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(normalized_reward, copy=False))  # Use normalized reward
            batch_dones.append(np.array(done, copy=False))

        return (
            torch.FloatTensor(np.array(batch_states)).to(device),
            torch.FloatTensor(np.array(batch_actions)).to(device),
            torch.FloatTensor(np.array(batch_next_states)).to(device),
            torch.FloatTensor(np.array(batch_rewards)).to(device).unsqueeze(1),
            torch.FloatTensor(np.array(batch_dones)).to(device).unsqueeze(1)
        )

# TD3 Agent
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)  # Reduced from 3e-4

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)  # Reduced from 3e-4

        self.max_action = max_action
        self.action_dim = action_dim

        self.total_it = 0
        self.exploration_noise = 0.1
        self.exploration_decay = 0.9999
        self.training = True  # Add this line

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if self.training:
            action = action + np.random.normal(0, self.exploration_noise, size=self.action_dim)
            self.exploration_noise *= self.exploration_decay
            self.exploration_noise = max(0.01, self.exploration_noise)  # Set a minimum exploration noise
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=100,
              discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # with torch.no_grad():
            # Select action according to policy and add clipped noise
        noise = (
            torch.randn_like(action) * policy_noise
        ).clamp(-noise_clip, noise_clip)

        next_action = (
            self.actor_target(next_state) + noise
        ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1-done) * discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename="model"):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename="model"):
        try:
            self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
            self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing model found. Starting with a new model.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False
