import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
from utils import get_config
from networks import Hybrid_SAC_Actor


class Hybrid_DSAC_Critic(nn.Module):
    """
    DSAC-T style distributional critic for hybrid actions.

    Input:
        state:              (B, state_size)
        continuous_action:  (B, continuous_action_size)
        discrete_onehot:    (B, discrete_action_size)

    Output:
        q_mean:             (B, 1)
        q_std:              (B, 1), positive
    """
    def __init__(self, state_size, continuous_action_size, discrete_action_size, layer_size=256):
        super().__init__()
        input_size = state_size + continuous_action_size + discrete_action_size

        self.net = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 2),
        )

    def forward(self, states, continuous_actions, discrete_onehot_actions):
        x = torch.cat([states, continuous_actions, discrete_onehot_actions], dim=-1)
        out = self.net(x)
        q_mean, q_std_raw = torch.chunk(out, chunks=2, dim=-1)

        # DSAC-v2 的 ActionValueDistri 也是使用 softplus 讓 std 為正
        # 這裡加上一個很小的 epsilon，避免除以 0
        q_std = F.softplus(q_std_raw) + 1e-6
        return q_mean, q_std


class SACAgent():
    """
    Online Independent Hybrid DSAC-T agent.

    This is a DSAC-T-inspired modification of the original Hybrid SAC agent:
      - Actor remains a hybrid stochastic policy:
            continuous movement: tanh Gaussian
            discrete device selection: categorical distribution
      - Critic is changed from scalar Q(s,a) to Gaussian value distribution:
            Z(s,a) = Normal(Q(s,a), sigma(s,a)^2)
      - Critic update uses DSAC-T style components:
            expected value substitution
            twin value distribution learning
            variance-based critic gradient adjustment

    Note:
      This is still online off-policy RL, not offline RL and not conservative RL.
    """
    def __init__(self, state_size, continuous_action_size, discrete_action_size, hidden_size=256, device="cpu"):
        config = get_config()

        self.state_size = state_size
        self.continuous_action_size = continuous_action_size
        self.discrete_action_size = discrete_action_size
        self.device = device

        # Soft update
        self.tau = getattr(config, "tau", 1e-3)

        # Discount factor
        self.gamma = getattr(config, "gamma", 0.99)

        # Entropy coefficient
        # 先沿用原始 Hybrid SAC 的固定 alpha
        self.alpha = getattr(config, "alpha", 0.2)

        # Learning rate
        self.lr = getattr(config, "lr", 1e-4)

        # Actor log_std clamp
        self.log_std_min = -20
        self.log_std_max = 2

        self.max_mov = config.max_mov

        # DSAC-T hyperparameters
        self.dsac_xi = getattr(config, "dsac_xi", 3.0)
        self.dsac_eps = getattr(config, "dsac_eps", 1e-6)
        self.dsac_omega_eps = getattr(config, "dsac_omega_eps", 1e-6)
        self.dsac_initial_b = getattr(config, "dsac_initial_b", 20.0)
        self.dsac_initial_omega = getattr(config, "dsac_initial_omega", 1.0)

        self.actor_network = Hybrid_SAC_Actor(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)

        # DSAC-T paper uses target policy network in the target distribution.
        # This is different from many SAC implementations, but closer to DSAC-T.
        self.target_actor_network = Hybrid_SAC_Actor(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network_1 = Hybrid_DSAC_Critic(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)

        self.critic_network_2 = Hybrid_DSAC_Critic(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)

        self.target_critic_network_1 = Hybrid_DSAC_Critic(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)

        self.target_critic_network_2 = Hybrid_DSAC_Critic(
            state_size=self.state_size,
            continuous_action_size=self.continuous_action_size,
            discrete_action_size=self.discrete_action_size,
            layer_size=hidden_size,
        ).to(self.device)

        self.target_critic_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.target_critic_network_2.load_state_dict(self.critic_network_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_network_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_network_2.parameters(), lr=self.lr)

        # Moving averages for DSAC-T variance-based critic gradient adjustment
        self.b_1 = torch.tensor(float(self.dsac_initial_b), device=self.device)
        self.b_2 = torch.tensor(float(self.dsac_initial_b), device=self.device)
        self.omega_1 = torch.tensor(float(self.dsac_initial_omega), device=self.device)
        self.omega_2 = torch.tensor(float(self.dsac_initial_omega), device=self.device)

    def sample_policy(self, states, deterministic=False, actor_network=None):
        if actor_network is None:
            actor_network = self.actor_network

        continuous_mu, continuous_log_std, discrete_logits = actor_network(states)

        continuous_log_std = torch.clamp(continuous_log_std, self.log_std_min, self.log_std_max)
        continuous_std = continuous_log_std.exp()

        normal_dist = Normal(continuous_mu, continuous_std)

        if deterministic:
            raw_continuous_action = continuous_mu
        else:
            raw_continuous_action = normal_dist.rsample()

        tanh_action = torch.tanh(raw_continuous_action)
        continuous_action = tanh_action * self.max_mov

        continuous_log_prob = normal_dist.log_prob(raw_continuous_action) - torch.log(
            1.0 - tanh_action.pow(2) + 1e-6
        )

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
        return F.one_hot(discrete_action, num_classes=self.discrete_action_size).float()

    def critic_all_discrete(self, critic, states, continuous_actions):
        """
        Enumerate all discrete actions for a fixed continuous action.

        Return:
            q_mean_all: (B, D)
            q_std_all:  (B, D)
        """
        batch_size = states.shape[0]
        D = self.discrete_action_size

        states_expanded = states.unsqueeze(1).expand(batch_size, D, self.state_size)
        continuous_expanded = continuous_actions.unsqueeze(1).expand(
            batch_size, D, self.continuous_action_size
        )

        discrete_indices = torch.arange(D, device=self.device).unsqueeze(0).expand(batch_size, D)
        discrete_onehot = F.one_hot(discrete_indices, num_classes=D).float()

        states_flat = states_expanded.reshape(batch_size * D, self.state_size)
        continuous_flat = continuous_expanded.reshape(batch_size * D, self.continuous_action_size)
        discrete_onehot_flat = discrete_onehot.reshape(batch_size * D, D)

        q_mean_flat, q_std_flat = critic(states_flat, continuous_flat, discrete_onehot_flat)

        q_mean_all = q_mean_flat.view(batch_size, D)
        q_std_all = q_std_flat.view(batch_size, D)

        return q_mean_all, q_std_all

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.float().to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.actor_network.eval()

        with torch.no_grad():
            continuous_action, _, discrete_action, _, _ = self.sample_policy(
                state, deterministic=deterministic, actor_network=self.actor_network
            )

        self.actor_network.train()

        continuous_action = continuous_action.cpu().numpy()[0]
        discrete_action = int(discrete_action.cpu().numpy()[0])

        return [discrete_action, continuous_action]

    def _update_running_b_omega(self, idx, current_std):
        """Update moving averages b_i and omega_i for each critic."""
        with torch.no_grad():
            std_mean = current_std.detach().mean()
            std2_mean = current_std.detach().pow(2).mean()

            if idx == 1:
                self.b_1 = self.tau * self.dsac_xi * std_mean + (1.0 - self.tau) * self.b_1
                self.omega_1 = self.tau * std2_mean + (1.0 - self.tau) * self.omega_1
            elif idx == 2:
                self.b_2 = self.tau * self.dsac_xi * std_mean + (1.0 - self.tau) * self.b_2
                self.omega_2 = self.tau * std2_mean + (1.0 - self.tau) * self.omega_2
            else:
                raise ValueError(f"Unknown critic index: {idx}")

    def _dsac_t_critic_loss(self, current_q, current_std, y_q, y_z, idx):
        """
        DSAC-T-style critic objective.

        This proxy loss is constructed so that its gradients match the DSAC-T
        mean-related and variance-related gradient directions:
          - Expected value substitution for mean Q update
          - Random target return for variance update
          - Variance-based clipping boundary and gradient scaling
        """
        self._update_running_b_omega(idx, current_std)

        if idx == 1:
            b = self.b_1.detach()
            omega = self.omega_1.detach()
        else:
            b = self.b_2.detach()
            omega = self.omega_2.detach()

        # Mean-related part:
        # Use expected target y_q instead of random target y_z.
        mean_denom = current_std.detach().pow(2) + self.dsac_eps
        mean_loss = 0.5 * (current_q - y_q.detach()).pow(2) / mean_denom

        # Variance-related part:
        # Clip the random target return around current Q to stabilize std learning.
        clipped_y_z = torch.max(
            torch.min(y_z.detach(), current_q.detach() + b),
            current_q.detach() - b,
        )
        variance_error = clipped_y_z - current_q.detach()
        std_safe = current_std + self.dsac_eps
        variance_loss = torch.log(std_safe) + 0.5 * variance_error.pow(2) / std_safe.pow(2)

        scale = omega + self.dsac_omega_eps
        critic_loss = scale * (mean_loss + variance_loss).mean()
        return critic_loss

    def Learn_DSAC_T(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = states.float().to(self.device)
        actions = actions.float().to(self.device)
        rewards = rewards.float().view(-1, 1).to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.float().view(-1, 1).to(self.device)

        # ReplayBuffer may return shape (B, 1, action_dim).
        if actions.dim() == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        discrete_actions = actions[:, 0].long()
        continuous_actions = actions[:, 1: 1 + self.continuous_action_size]
        discrete_actions_onehot = self.discrete_to_onehot(discrete_actions).to(self.device)

        # -------------------------
        # Build DSAC-T targets
        # -------------------------
        with torch.no_grad():
            next_continuous_actions, next_continuous_log_prob, _, next_discrete_probs, next_discrete_log_probs = self.sample_policy(
                next_states,
                deterministic=False,
                actor_network=self.target_actor_network,
            )

            target_q1_all, target_std1_all = self.critic_all_discrete(
                self.target_critic_network_1,
                next_states,
                next_continuous_actions,
            )
            target_q2_all, target_std2_all = self.critic_all_discrete(
                self.target_critic_network_2,
                next_states,
                next_continuous_actions,
            )

            # Twin value distribution learning:
            # choose distribution with lower mean Q for each discrete action.
            use_q1 = target_q1_all <= target_q2_all
            target_min_q_all = torch.where(use_q1, target_q1_all, target_q2_all)
            target_min_std_all = torch.where(use_q1, target_std1_all, target_std2_all)

            # Random target return sample from the chosen value distribution.
            target_z_all = target_min_q_all + target_min_std_all * torch.randn_like(target_min_std_all)

            # Hybrid-action entropy term.
            # next_continuous_log_prob: (B, 1), broadcasts to (B, D)
            entropy_all = self.alpha * next_discrete_log_probs + self.alpha * next_continuous_log_prob

            next_soft_q_all = target_min_q_all - entropy_all
            next_soft_z_all = target_z_all - entropy_all

            # Expectation over all discrete actions under π_d(.|s').
            next_soft_value_q = (next_discrete_probs * next_soft_q_all).sum(dim=1, keepdim=True)
            next_soft_value_z = (next_discrete_probs * next_soft_z_all).sum(dim=1, keepdim=True)

            y_q = rewards + self.gamma * (1.0 - dones) * next_soft_value_q
            y_z = rewards + self.gamma * (1.0 - dones) * next_soft_value_z

        # -------------------------
        # Critic 1 update
        # -------------------------
        current_q1, current_std1 = self.critic_network_1(
            states,
            continuous_actions,
            discrete_actions_onehot,
        )
        critic_1_loss = self._dsac_t_critic_loss(current_q1, current_std1, y_q, y_z, idx=1)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # -------------------------
        # Critic 2 update
        # -------------------------
        current_q2, current_std2 = self.critic_network_2(
            states,
            continuous_actions,
            discrete_actions_onehot,
        )
        critic_2_loss = self._dsac_t_critic_loss(current_q2, current_std2, y_q, y_z, idx=2)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # -------------------------
        # Actor update
        # -------------------------
        new_continuous_actions, continuous_log_prob, _, discrete_probs, discrete_log_probs = self.sample_policy(
            states,
            deterministic=False,
            actor_network=self.actor_network,
        )

        q1_all, _ = self.critic_all_discrete(self.critic_network_1, states, new_continuous_actions)
        q2_all, _ = self.critic_all_discrete(self.critic_network_2, states, new_continuous_actions)
        min_q_all = torch.min(q1_all, q2_all)

        # DSAC-T still uses mean Q for policy improvement by default.
        actor_loss_all = self.alpha * continuous_log_prob + self.alpha * discrete_log_probs - min_q_all
        actor_loss = (discrete_probs * actor_loss_all).sum(dim=1, keepdim=True).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------------
        # Target network update
        # -------------------------
        self.soft_update(self.critic_network_1, self.target_critic_network_1)
        self.soft_update(self.critic_network_2, self.target_critic_network_2)
        self.soft_update(self.actor_network, self.target_actor_network)

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item(),
            "b_1": float(self.b_1.detach().cpu().item()),
            "b_2": float(self.b_2.detach().cpu().item()),
            "omega_1": float(self.omega_1.detach().cpu().item()),
            "omega_2": float(self.omega_2.detach().cpu().item()),
            "std_1_mean": float(current_std1.detach().mean().cpu().item()),
            "std_2_mean": float(current_std2.detach().mean().cpu().item()),
        }

    # Keep compatibility with old training loop.
    def Learn_SAC(self, experiences):
        return self.Learn_DSAC_T(experiences)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_parameter(self):
        return {
            "model": "agent_Online_DSAC_T",
            "tau": self.tau,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "lr": self.lr,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "dsac_xi": self.dsac_xi,
            "dsac_eps": self.dsac_eps,
            "dsac_omega_eps": self.dsac_omega_eps,
            "dsac_initial_b": self.dsac_initial_b,
            "dsac_initial_omega": self.dsac_initial_omega,
        }
