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
    def __init__(self, max_size=1e6, 
                 positive_sample_ratio = 0.5, postive_sample_threshold = 0):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.positive_sample_ratio = positive_sample_ratio
        self.postive_sample_threshold = postive_sample_threshold

    def add(self, transition):
        if ((transition[-2] >= self.postive_sample_threshold) or 
            (random.random() < (1 - self.positive_sample_ratio))):
            if len(self.storage) == self.max_size:
                self.storage[int(self.ptr)] = transition
                self.ptr = (self.ptr + 1) % self.max_size
            else:
                self.storage.append(transition)
                self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)      
        # storage_copy = copy.deepcopy(self.storage)
        # np.random.shuffle(storage_copy)
        # # Get indices of tuples where the last element is positive
        # pos_indices = [i for i, t in enumerate(storage_copy) if t[-2] >= self.postive_sample_threshold]
        # # print(f"postive samples:{len(pos_indices)}")
        # if int(batch_size * self.positive_sample_ratio) < len(pos_indices):
        #     pos_ind_ind = np.random.randint(0, len(pos_indices), size = int(batch_size * self.positive_sample_ratio))
        #     pos_ind =  [pos_indices[i] for i in pos_ind_ind]
        # else: pos_ind = pos_indices
        # remaining_list_ind = [i for i, t in enumerate(storage_copy) if i not in pos_ind]
        # rand_ind_ind = np.random.randint(0, len(remaining_list_ind), size = batch_size - len(pos_ind))
        # rand_ind = np.array([remaining_list_ind[i] for i in rand_ind_ind])
        
        # ind = np.concatenate((pos_ind ,rand_ind))
        # np.random.shuffle(ind)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            # state, action, next_state, reward, done = storage_copy[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        # print(torch.FloatTensor(np.array(batch_states)).to(device).size())
        # print(torch.FloatTensor(np.array(batch_actions)).to(device).size())
        # print(torch.FloatTensor(np.array(batch_next_states)).to(device).size())
        # print(torch.FloatTensor(np.array(batch_rewards)).to(device).unsqueeze(1).size())
        # print(torch.FloatTensor(np.array(batch_dones)).to(device).unsqueeze(1).size())
        
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        # self.discount = 0.99
        # self.tau = 0.005

        # self.policy_noise = 0.2
        # self.noise_clip = 0.5
        # self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

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
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
