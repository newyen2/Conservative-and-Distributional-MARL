import torch
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_


# 從動作集隨機採樣，重複100次並儲存至資料集
def collect_random(env, U, dataset, num_samples=100):
    state = env.reset()
    for _ in range(num_samples):
        action = random.sample(range(0, 5*(env.M+1)), U)
        next_state, reward, done = env.step(state,action)
        
        for u in range(U):
            dataset[u].add(state, action[u], reward[u], next_state, done[u])
            
        state = next_state
        if env.DONE:
            state = env.reset()

# 取得設定      
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--episodes", type=int, default=200) # Online的episode次數, default: 200
    parser.add_argument("--epochs", type=int, default=150) # Offline的epoch次數, default: 150
    parser.add_argument("--buffer_size", type=int, default=100_000) # Online的Replay Buffer大小, default: 100_000
    parser.add_argument("--seed", type=int, default=1) # 隨機種子, default: 1
    parser.add_argument("--min_eps", type=float, default=0.01) # Online-DQN的最小探索epsilon, default: 0.01
    parser.add_argument("--eps_frames", type=int, default=1e4) # 衰減至min_eps所需的步數, default: 1e4
    parser.add_argument("--Batch_online", type=int, default=32) # Online的Batch大小, default: 32
    parser.add_argument("--Batch_offline", type=int, default=128) # Offline的Batch大小, default: 128
    parser.add_argument("--eval_every", type=int, default=1) # 評估當前模型的頻率, default: 1
    parser.add_argument("--M", type=int, default=10) # Device的數量, default: 10
    parser.add_argument("--Num_Cells", type=int, default=10) # 環境的格子邊長, default: 10
    parser.add_argument("--U", type=int, default=2) # Agent的數量, default: 2
    parser.add_argument("--DELTA", type=int, default=500) # 獎勵函數的能源權重,在論文內使用的是lambda, default: 500
    parser.add_argument("--penalty", type=int, default=300) # 風險產生的懲罰值, default: 300
    parser.add_argument("--data_size_perc", type=int, default=16) # 從離線資料集用於訓練的資料比例, default: 16
    parser.add_argument("--prob_of_risk", type=int, default=0.1) # 具有風險時的懲罰機率, default: 0.1
    parser.add_argument("--PATH", type=str, default=r"C:\Users\wish1\Desktop\Conservative-and-Distributional-MARL-main\\") # 存檔路徑
    
    return parser.parse_args(args=[])

# 模擬100次的執行並取得獎勵
# epsilon = 0
def eval_runs(env, agent, eval_runs=100):
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        action = [0] * env.U
        while True:
            for u in range(env.U):
                action_agent = agent[u].get_action(state, 0)
                action[u] = action_agent[0]
                
            next_state, reward, done = env.step(state,action)
            
            rewards += env.Total_reward
            state = next_state
            if env.DONE:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

# 模擬100次的執行並取得獎勵
# distributional版本，由於get_action回傳結構不同
# epsilon = 0
def eval_runs_dist(env, agent, eval_runs=100):
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        action = [0] * env.U
        while True:
            for u in range(env.U):
                action_agent = agent[u].get_action(state, 0)
                action[u] = action_agent
                
            next_state, reward, done = env.step(state,action)
            
            rewards += env.Total_reward
            state = next_state
            if env.DONE:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

# 預處理dataset，並轉換成(s, a, r, s′, done)的形式，再轉成DataLoader
def prep_dataloader(state_dim,dataset,num_UAVs,batch_size=256):
    
    state_start = str(0)
    state_end = str(state_dim - 1)
    action_start = str(state_dim)
    action_end = str(state_dim + num_UAVs - 1)
    reward_col = str(state_dim + num_UAVs)
    next_state_start = str(state_dim + num_UAVs + 1)
    next_state_end = str(state_dim + num_UAVs + state_dim)
    done_col = str(state_dim + num_UAVs + state_dim + 1)    
    
    states_df = dataset.loc[:, state_start : state_end]
    actions_df = dataset.loc[:, action_start:action_end]
    rewards_df = dataset.loc[:,reward_col]
    next_states_df = dataset.loc[:, next_state_start : next_state_end]
    done_df = dataset.loc[:,done_col]

    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.long)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    
    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])
    
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader

# 集中式Q-Network更新
def loss_update_cent(q1_loss, Net_params, Tar_params,optimizer):
    optimizer.zero_grad()
    q1_loss.backward()
    clip_grad_norm_(Net_params, 1.)
    optimizer.step()

    soft_update_cent(Net_params, Tar_params, tau=1e-3)
    return q1_loss.detach().item()

# 對目標網路進行Soft Update
def soft_update_cent(Net_params, Tar_params, tau):
    for target_param, local_param in zip(Tar_params, Net_params):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

