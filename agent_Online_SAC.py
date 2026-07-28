import torch
from networks import Hybrid_SAC_Actor, Hybrid_SAC_Critic
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
from utils import get_config

class SACAgent():
    def __init__(self, state_size, continuous_action_size, discrete_action_size, hidden_size=256, device="cpu"):
        # TODO(HRL): 後續需初始化 high DQN、MoveActor、SelectActor、兩個 joint Critic 及各自 target/optimizer。
        # 目前保留 Hybrid SAC 初始化，確保尚未接上新 Trainer 前原本程式仍可執行。
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

        continuous_action = tanh_action * 5 + 5

        continuous_log_prob = normal_dist.log_prob(raw_continuous_action) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)

        scale = torch.as_tensor(5, dtype=torch.float32, device=self.device)
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
        # TODO(HRL): 此方法仍回傳舊式合併動作；階層式 Trainer 應分別呼叫 goal、move 與 service 介面。
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
        # TODO(HRL): 此為舊 Hybrid SAC 更新；階層式版本需拆成 Learn_High 與 Learn_Low。
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

    def initialize_hierarchical_networks(self, hidden_size=256):
        """新增理由：將階層式網路建立集中在單一入口，且不直接移除目前 Hybrid SAC 初始化流程。"""
        pass

    def goal_to_onehot(self, goals):
        """新增理由：將高層選定的裝置索引轉成 M 維條件向量，而不是把 DQN 的 Q-values 當成 goal。"""
        pass

    def select_high_goal(self, state, epsilon=0.0, deterministic=False):
        """新增理由：高層 DQN 需以 epsilon-greedy 選擇目標，並由 Trainer 在 n 步內持續沿用。"""
        pass

    def sample_move(self, states, goals, deterministic=False):
        """新增理由：MoveActor 應只根據 S 與高層 goal 產生受移動上限約束的連續動作。"""
        pass

    def sample_service(self, moved_states, goals, action_mask=None, deterministic=False):
        """新增理由：SelectActor 必須在移動完成取得 S' 後，才依 goal 與合法動作遮罩選擇服務目標。"""
        pass

    def build_joint_critic_input(self, states, goals, move_actions, service_actions):
        """新增理由：統一拼接 S、goal、move 與 service one-hot，避免兩個 actor 對 Critic 輸入定義不一致。"""
        pass

    def Learn_High(self, experiences):
        """新增理由：高層需以獨立 DQN/target DQN 更新，並處理每段 goal duration 對應的折扣。"""
        pass

    def Learn_Low(self, experiences):
        """新增理由：低層兩個 actor 需透過相同 joint twin Critic 的 Q-value 協同更新。"""
        pass

    # Soft-Update目標網路     
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_parameter(self):
        # TODO(HRL): 後續需加入 high/low learning rate、目標更新、熵係數與 high interval 等參數。
        return {
            "model": "agent_Online_SAC",
            "tau": self.tau,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "lr": self.lr,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max
        }
