import torch
from networks import CQR_DQN
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils import get_config
import numpy as np
import random

class CQRAgent():
    def __init__(self, state_size, action_size, alpha, eta, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # 目標網路Soft Update
        self.tau = 1e-3

        # 折扣回報
        self.gamma = 0.99

        # 分位數數量
        self.N = 8

        # CQL權重
        self.alpha = alpha

        # 分位數位置
        self.eta = eta
        self.quantile_tau = self.eta * torch.FloatTensor([(2 * i + 1) / (2.0 * self.N) for i in range(0,self.N)]).to(device)
        
        self.network = CQR_DQN(state_size = self.state_size,
                    action_size = self.action_size,
                    layer_size = hidden_size,
                    N = self.N
                    ).to(self.device)
        self.target_network = CQR_DQN(state_size = self.state_size,
                    action_size = self.action_size,
                    layer_size = hidden_size,
                    N = self.N
                    ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

    # 根據epsilon-greedy取得動作
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            self.network.eval()
            with torch.no_grad():
                action_values = self.network.get_action(state)
            self.network.train()

            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choices(np.arange(self.action_size))
        return action

    # 計算CQL Loss  
    def CQL_Loss_Calc(self, q_values, current_action):
        logsumexp = torch.logsumexp(q_values, dim=2, keepdim=True)
        q_a = q_values.gather(2, current_action.repeat(1, self.N).unsqueeze(-1))

        return (logsumexp - q_a).sum(dim=1).mean()

    # 計算Huber Loss
    def Huber_Loss_Calc(self, td_error, kappa=1.0):
        loss = torch.where(td_error.abs() <= kappa, 0.5 * td_error.pow(2), kappa * (td_error.abs() - 0.5 * kappa))
        return loss

    # 計算Loss        
    def Loss_Calc(self, CQL_loss, Qt_expected, Qt_targets):

        # 計算TD誤差，每個預測分位數都與所有目標分位數計算誤差
        # (B, 1, N) - (B, N, 1) 
        td_error = Qt_targets - Qt_expected

        # 計算Huber Loss
        huber_loss = self.Huber_Loss_Calc(td_error, kappa = 1.0)

        # 計算Quantile Loss
        # TD Error >= 0 (低估)，權重為τ
        # TD Error < 0 (高估)，權重為(1-τ)
        loss_weight = abs(self.quantile_tau - (td_error.detach() < 0).float())
        quantile_loss = loss_weight * huber_loss
        
        # 先計算對於每個目標分位數的Loss總和，再對所有目標分位數取平均，最後對批次取平均
        quantile_loss = quantile_loss.sum(dim=1).mean(dim=1).mean()

        # 計算Loss
        # 前項為CQL項，後項為Quantile項
        # 僅保留後項則退化為QR-DQN
        loss = self.alpha * CQL_loss + 0.5 * quantile_loss

        return loss

    # 取得目標Quantile與預測Quantile
    def get_Qt_value(self, experiences, U):
        # 拆分經驗樣本
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # 取得批次下狀態的所有動作的分位數估計
            Qt_targets_next = self.target_network(next_states).detach() # shape: (B, N, A)

            # 對分位數估計取平均
            Q_targets_next = Qt_targets_next.mean(dim=1) # shape: (B, A)

            # 取得平均分位數估計最大的動作
            actions_expected = torch.argmax(Q_targets_next, dim=1, keepdim=True) # Shape: (B, 1)

            # 取得批次下狀態最佳動作的分位數估計
            Qt_targets_next = Qt_targets_next.gather(2, actions_expected.repeat(1, self.N).unsqueeze(-1)) # Shape: (B, N, 1)

            # Reshape以便後續Loss計算
            Qt_targets_next = Qt_targets_next.transpose(1,2) # Shape: (B, 1, N)

            # 計算目標Q值
            Qt_targets = (rewards / U).unsqueeze(-1) + (self.gamma * Qt_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))

        # 取得期望Q值
        Qt_expected = self.network(states).gather(2, actions.repeat(1, self.N).unsqueeze(-1)) # Shape: (B, N, 1)

        return Qt_targets, Qt_expected

    def Learn_CQR_ind(self, experiences):
        # 拆分經驗樣本
        states, actions, _, _, _ = experiences

        # 取得目標Quantile與預測Quantile
        Qt_targets, Qt_expected = self.get_Qt_value(experiences, U = 1)

        # 取得該狀態所有動作的Quantile
        Qt_a_s = self.network(states)

        # 計算CQL Loss
        CQL_loss = self.CQL_Loss_Calc(Qt_a_s, actions)

        # 計算Loss
        loss = self.Loss_Calc(CQL_loss, Qt_expected, Qt_targets)
        
        # 更新預測網路
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # 更新目標網路
        self.soft_update(self.network, self.target_network)

        return loss.detach().item()
    
    def Learn_CQR_cent(self, experiences):
        config = get_config()

        # 拆分經驗樣本
        states, actions, _, _, _ = experiences

        # 取得目標Quantile與預測Quantile
        Qt_targets, Qt_expected = self.get_Qt_value(experiences, U = config.U)
    
        # 取得該狀態所有動作的Quantile
        Qt_a_s = self.network(states)

        # 計算CQL Loss
        CQL_loss = self.CQL_Loss_Calc(Qt_a_s, actions)
        
        # 回傳至MA_CCQR進行聚合
        return CQL_loss, Qt_expected, Qt_targets

    # Soft-Update目標網路
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
