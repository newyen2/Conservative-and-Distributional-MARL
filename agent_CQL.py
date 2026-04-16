import torch
from networks import DDQN
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import get_config
import numpy as np
import random

class CQLAgent():
    def __init__(self, state_size, action_size, alpha, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # 目標網路Soft Update
        self.tau = 1e-3

        # 折扣回報
        self.gamma = 0.99

        # CQL權重
        self.alpha = alpha
        
        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-4)

    # 根據epsilon-greedy取得動作
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()

            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action

    # 計算CQL Loss  
    def CQL_Loss_Calc(self, q_values, current_action):
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()

    # 計算Loss
    def Loss_Calc(self, CQL_loss, Q_expected, Q_targets):

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        loss = self.alpha * CQL_loss + 0.5 * bellman_error
        return loss

    # 取得目標Quantile與預測Quantile
    def get_Qt_value(self, experiences, U):
        # 拆分經驗樣本
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards / U + (self.gamma * Q_targets_next * (1 - dones))

        # 取得該狀態的所有動作Q值
        Q_a_s = self.network(states)

        # 取得期望Q值
        Q_expected = Q_a_s.gather(1, actions)

        return Q_targets, Q_expected
    
    def Learn_CQL_ind(self, experiences): 
        # 拆分經驗樣本
        states, actions, _, _, _ = experiences
        
        # 取得目標Q值與預測Q值
        Q_targets, Q_expected = self.get_Qt_value(experiences, U = 1)

        # 取得該狀態的所有動作Q值
        Q_a_s = self.network(states)

        # 計算CQL Loss
        CQL_loss = self.CQL_Loss_Calc(Q_a_s, actions)

        # 計算Loss
        loss = self.Loss_Calc(CQL_loss, Q_expected, Q_targets)

        # 更新預測網路
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # 更新目標網路
        self.soft_update(self.network, self.target_network)

        return loss.detach().item()
   
    def Learn_CQL_cent(self, experiences):
        config = get_config()

        # 拆分經驗樣本
        states, actions, _, _, _ = experiences
        
        # 取得目標Q值與預測Q值
        Q_targets, Q_expected = self.get_Qt_value(experiences, U = config.U)

        # 取得該狀態的所有動作Q值
        Q_a_s = self.network(states)

        # 計算CQL Loss
        CQL_loss = self.CQL_Loss_Calc(Q_a_s, actions)
        
        # 回傳至MA_CCQL進行聚合
        return CQL_loss, Q_expected, Q_targets
    
    # Soft-Update目標網路
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

