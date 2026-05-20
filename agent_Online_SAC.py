import torch
from networks import Hybrid_SAC_Actor, Hybrid_SAC_Critic
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
from utils import get_config

class SACAgent():
    def __init__(self, state_size, continuous_action_size, discrete_action_size, hidden_size=256, device="cpu"):
        config = get_config()

        self.state_size = state_size
        self.continuous_action_size = continuous_action_size
        self.discrete_action_size = discrete_action_size
        self.device = device

        # 目標網路Soft Update
        self.tau = 1e-3

        # 折扣回報
        self.gamma = 0.99      

        # Entropy
        self.alpha = 0.2

        # Learning Rate
        self.lr = 1e-4

        # log_std clamp 範圍
        self.log_std_min = -20
        self.log_std_max = 2

        self.max_mov = config.max_mov

        self.actor_network = Hybrid_SAC_Actor(state_size=self.state_size,
                                              continuous_action_size=self.continuous_action_size,
                                              discrete_action_size=self.discrete_action_size,
                                              layer_size=hidden_size).to(self.device)
        
        self.critic_network_1 = Hybrid_SAC_Critic(state_size=self.state_size,
                                              continuous_action_size=self.continuous_action_size,
                                              discrete_action_size=self.discrete_action_size,
                                              layer_size=hidden_size).to(self.device)
        
        self.critic_network_2 = Hybrid_SAC_Critic(state_size=self.state_size,
                                              continuous_action_size=self.continuous_action_size,
                                              discrete_action_size=self.discrete_action_size,
                                              layer_size=hidden_size).to(self.device)
        
        self.target_critic_network_1 = Hybrid_SAC_Critic(state_size=self.state_size,
                                              continuous_action_size=self.continuous_action_size,
                                              discrete_action_size=self.discrete_action_size,
                                              layer_size=hidden_size).to(self.device)
        
        self.target_critic_network_2 = Hybrid_SAC_Critic(state_size=self.state_size,
                                              continuous_action_size=self.continuous_action_size,
                                              discrete_action_size=self.discrete_action_size,
                                              layer_size=hidden_size).to(self.device)

        self.target_critic_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.target_critic_network_2.load_state_dict(self.critic_network_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_network_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_network_2.parameters(), lr=self.lr)

    def sample_policy(self, states, deterministic=False):
        continuous_mu, continuous_log_std, discrete_logits = self.actor_network(states)

        continuous_log_std = torch.clamp(continuous_log_std, self.log_std_min, self.log_std_max)
        continuous_std = continuous_log_std.exp()

        normal_dist = Normal(continuous_mu, continuous_std)

        if deterministic:
            raw_continuous_action = continuous_mu
        else:
            raw_continuous_action = normal_dist.rsample()

        tanh_action = torch.tanh(raw_continuous_action)

        continuous_action = tanh_action * self.max_mov 

        continuous_log_prob = normal_dist.log_prob(raw_continuous_action) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)

        scale = torch.as_tensor(self.max_mov, dtype=torch.float32, device=self.device)
        continuous_log_prob -= torch.log(scale + 1e-6)

        continuous_log_prob = continuous_log_prob.sum(dim=-1, keepdim=True)

        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_log_probs = F.log_softmax(discrete_logits, dim=-1)

        discrete_dist = Categorical(probs=discrete_probs)

        if deterministic:
            discrete_action = torch.argmax(discrete_probs, dim=-1)
        else:
            discrete_action = discrete_dist.sample()

        return continuous_action, continuous_log_prob, discrete_action, discrete_probs, discrete_log_probs
    
    def discrete_to_onehot(self, discrete_action):
        discrete_action = discrete_action.long()

        return F.one_hot(
            discrete_action,
            num_classes=self.discrete_action_size
        ).float()

    def critic_all_discrete(self, critic, states, continuous_actions):
        batch_size = states.shape[0]
        D = self.discrete_action_size

        states_expanded = states.unsqueeze(1).expand(batch_size, D, self.state_size)

        continuous_expanded = continuous_actions.unsqueeze(1).expand(batch_size, D, self.continuous_action_size)

        discrete_indices = torch.arange(D, device=self.device).unsqueeze(0).expand(batch_size, D)

        discrete_onehot = F.one_hot(discrete_indices, num_classes=D).float()

        states_flat = states_expanded.reshape(batch_size * D, self.state_size)

        continuous_flat = continuous_expanded.reshape(batch_size * D, self.continuous_action_size)

        discrete_onehot_flat = discrete_onehot.reshape(batch_size * D, D)

        q_flat = critic(states_flat, continuous_flat, discrete_onehot_flat)

        q_values = q_flat.view(batch_size, D)

        return q_values
    
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.float().to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.actor_network.eval()

        with torch.no_grad():
            continuous_action, _, discrete_action, _, _ = self.sample_policy(state, deterministic=deterministic)

        self.actor_network.train()

        continuous_action = continuous_action.cpu().numpy()[0]
        discrete_action = int(discrete_action.cpu().numpy()[0])

        return [discrete_action, continuous_action]
        
    def Learn_SAC(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = states.float().to(self.device)
        actions = actions.float().to(self.device)
        rewards = rewards.float().view(-1, 1).to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.float().view(-1, 1).to(self.device)

        # 如果 ReplayBuffer 存成 shape = (B, 1, action_dim)，這裡壓回 (B, action_dim)
        if actions.dim() == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        discrete_actions = actions[:, 0].long()
        continuous_actions = actions[:, 1 : 1 + self.continuous_action_size]

        discrete_actions_onehot = self.discrete_to_onehot(discrete_actions).to(self.device)

        with torch.no_grad():
            next_continuous_actions, next_continuous_log_prob, _, next_discrete_probs, next_discrete_log_probs = self.sample_policy(next_states, deterministic=False)

            target_q1_all = self.critic_all_discrete(self.target_critic_network_1, next_states, next_continuous_actions)

            target_q2_all = self.critic_all_discrete(self.target_critic_network_2, next_states, next_continuous_actions)

            target_min_q_all = torch.min(target_q1_all, target_q2_all)

            next_soft_q_all = target_min_q_all - self.alpha * next_discrete_log_probs - self.alpha * next_continuous_log_prob

            next_soft_value = (next_discrete_probs * next_soft_q_all).sum(dim=1, keepdim=True)

            q_targets = rewards + self.gamma * (1.0 - dones) * next_soft_value
        
        current_q1 = self.critic_network_1(states, continuous_actions, discrete_actions_onehot)
        current_q2 = self.critic_network_2(states, continuous_actions, discrete_actions_onehot)  

        critic_1_loss = F.mse_loss(current_q1, q_targets)
        critic_2_loss = F.mse_loss(current_q2, q_targets)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_continuous_actions, continuous_log_prob, _, discrete_probs, discrete_log_probs = self.sample_policy(states, deterministic=False)

        q1_all = self.critic_all_discrete(self.critic_network_1, states, new_continuous_actions)
        q2_all = self.critic_all_discrete(self.critic_network_2, states, new_continuous_actions)
        min_q_all = torch.min(q1_all, q2_all)

        actor_loss_all = self.alpha * continuous_log_prob + self.alpha * discrete_log_probs - min_q_all

        actor_loss = (discrete_probs * actor_loss_all).sum(dim=1, keepdim=True).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_network_1, self.target_critic_network_1)

        self.soft_update(self.critic_network_2, self.target_critic_network_2)

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item()
        }

    # Soft-Update目標網路     
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_parameter(self):
        return {
            "model": "agent_Online_SAC",
            "tau": self.tau,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "lr": self.lr,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max
        }