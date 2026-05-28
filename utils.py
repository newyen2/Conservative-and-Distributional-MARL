import torch
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_


# 從動作集隨機採樣，重複500次並儲存至資料集
def collect_random(env, U, dataset, steps = 500):
    config = get_config()
    state = env.reset()
    for _ in range(steps):

        actions = []

        for _ in range(U):
            select_action = random.randrange(env.nAction_select)
            move_action = np.random.uniform(
                low=-config.max_mov,
                high=config.max_mov,
                size=env.nAction_move
            ).astype(np.float32)

            actions.append([select_action, move_action])

        next_state, reward, done = env.step(state, actions)
        
        for u in range(U):
            action_vec = np.concatenate(([actions[u][0]], actions[u][1]), axis=None).astype(np.float32)
            dataset[u].add(state, action_vec, reward[u], next_state, done[u])

        state = next_state

        if env.DONE:
            state = env.reset()

# 取得設定      
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--episodes", type=int, default=200) # Online的episode次數, default: 200
    parser.add_argument("--epochs", type=int, default=150) # Offline的epoch次數, default: 150
    parser.add_argument("--buffer_size", type=int, default=100_000) # Online的Replay Buffer大小, default: 100_000
    parser.add_argument("--train_seed", type=int, default=1) # 訓練種子, default: 1
    parser.add_argument("--eval_seed", type=int, default=1) # 測試種子, default: 1
    parser.add_argument("--min_eps", type=float, default=0.01) # Online-DQN的最小探索epsilon, default: 0.01
    parser.add_argument("--eps_frames", type=int, default=1e4) # 衰減至min_eps所需的步數, default: 1e4
    parser.add_argument("--batch_size_online", type=int, default=32) # Online的Batch大小, default: 32
    parser.add_argument("--batch_size_offline", type=int, default=128) # Offline的Batch大小, default: 128
    parser.add_argument("--eval_periods", type=int, default=1) # 評估當前模型的頻率, default: 1
    parser.add_argument("--M", type=int, default=10) # Device的數量, default: 10
    parser.add_argument("--nCell", type=int, default=10) # 環境的格子邊長, default: 10
    parser.add_argument("--U", type=int, default=2) # Agent的數量, default: 2
    parser.add_argument("--energy_weight", type=int, default=500) # 獎勵函數的能源權重,在論文內使用的是lambda, default: 500
    parser.add_argument("--penalty", type=int, default=300) # 風險產生的懲罰值, default: 300
    parser.add_argument("--data_size", type=int, default=16) # 從離線資料集用於訓練的資料比例, default: 16
    parser.add_argument("--prob_of_risk", type=float, default=0.1) # 具有風險時的懲罰機率, default: 0.1
    parser.add_argument("--max_mov", type=float, default=2) # 每個時間步的最大位移(論文中steps_mov = 1)
    parser.add_argument("--boundary_penalty_weight", type=float, default=10.0) # 移動超出邊界時的懲罰值
    parser.add_argument("--max_power", type=float, default=10) # 能達成有效傳輸的最大功率
    parser.add_argument("--PATH", type=str, default=r"C:\Users\wish1\Desktop\Conservative-and-Distributional-MARL-main") # 存檔路徑
    
    return parser.parse_args(args=[])

# 模擬10次的執行並取得獎勵
def eval_runs(env, agent, rng, eval_runs=10):
    rng.switch_mode("eval")
    
    rewards = []

    for _ in range(eval_runs):
        state = env.reset()
        episode_reward = 0
        action = [0] * env.U

        while not env.DONE:
            for u in range(env.U):
                action[u] = agent[u].get_action(state, deterministic=True)
                
            next_state, _, _ = env.step(state, action)
            
            episode_reward += env.total_reward
            state = next_state

        rewards.append(episode_reward)

    rng.switch_mode("train")
    return np.mean(rewards)

# 預處理dataset，並轉換成(s, a, r, s′, done)的形式，再轉成DataLoader
def prep_dataloader(state_dim, dataset, num_UAVs, batch_size=256, continuous_action_size = 2):
    action_dim_per_uav = 1 + continuous_action_size
    total_action_dim = num_UAVs * action_dim_per_uav

    # 拆分dataset
    states_df = dataset.iloc[:, 1 : state_dim + 1]
    actions_df = dataset.iloc[:, state_dim + 1 : state_dim + total_action_dim + 1]
    rewards_df = dataset.iloc[:, state_dim + total_action_dim + 1]
    next_states_df = dataset.iloc[:, state_dim + total_action_dim + 2 : state_dim + total_action_dim + state_dim + 2]
    done_df = dataset.iloc[:, state_dim + total_action_dim + state_dim + 2]

    # 轉換為tensor
    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.float)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])
    
    # 轉換為DataLoader
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader

# 集中式Q-Network更新
def loss_update_cent(loss, Net_params, Tar_params,optimizer):
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(Net_params, 1.)
    optimizer.step()

    soft_update_cent(Net_params, Tar_params, tau=1e-3)
    return loss.detach().item()

# 對目標網路進行Soft Update
def soft_update_cent(Net_params, Tar_params, tau):
    for target_param, local_param in zip(Tar_params, Net_params):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# 隨機數管理
class RNGManager:
    def __init__(self):
        config = get_config()
        self.states = {}
  
        np.random.seed(config.train_seed)
        random.seed(config.train_seed)
        torch.manual_seed(config.train_seed)

        self.states["train"] = {"np": np.random.get_state(), 
                            "random": random.getstate(), 
                            "torch": torch.get_rng_state()}
        
        np.random.seed(config.eval_seed)
        random.seed(config.eval_seed)
        torch.manual_seed(config.eval_seed)

        self.states["eval"] = {"np": np.random.get_state(), 
                           "random": random.getstate(), 
                           "torch": torch.get_rng_state()}
        
        self.mode = "eval"

        self.switch_mode("train")

    def switch_mode(self, r_mode):
        if r_mode == self.mode: return
        if self.states[r_mode] == None: return

        self.states[self.mode] = {"np": np.random.get_state(), 
                           "random": random.getstate(), 
                           "torch": torch.get_rng_state()}

        np.random.set_state(self.states[r_mode]["np"])
        random.setstate(self.states[r_mode]["random"])
        torch.set_rng_state(self.states[r_mode]["torch"])

        self.mode = r_mode
    
    def get_state(self, r_mode):
        return self.states[r_mode]

    def force_set_state(self, r_mode ,r_states):
        self.states[r_mode] = r_states