import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import get_config
import pandas as pd
import torch.optim as optim
import numpy as np
from tqdm import tqdm

config = get_config()

# 預處理dataset，並轉換成(s, a, r, s′, done)的形式，再轉成DataLoader
def prep_dataloader(state_dim, dataset, num_UAVs, batch_size=256):
    
    # 拆分dataset
    states_df = dataset.iloc[:, 1 : state_dim + 1]
    actions_df = dataset.iloc[:, state_dim + 1 : state_dim + num_UAVs + 1]
    rewards_df = dataset.iloc[:,  state_dim + num_UAVs + 1]
    next_states_df = dataset.iloc[:, state_dim + num_UAVs + 2 : state_dim + num_UAVs + state_dim + 2]
    done_df = dataset.iloc[:, state_dim + num_UAVs + state_dim + 2]

    # 轉換為tensor
    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.long)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    
    state_mean = tensors["observations"].mean(dim=0, keepdim=True)
    state_std = tensors["observations"].std(dim=0, keepdim=True).clamp(min=1e-6)
    states = (tensors["observations"] - state_mean) / state_std

    next_state_mean = tensors["next_observations"].mean(dim=0, keepdim=True)
    next_state_std = tensors["next_observations"].std(dim=0, keepdim=True).clamp(min=1e-6)
    next_states = (tensors["next_observations"] - next_state_mean) / next_state_std

    action_onehots = []
    for u in range(num_UAVs):
        a_u = tensors["actions"][:, u]
        a_u_onehot = F.one_hot(a_u, num_classes=5 * (config.M + 1)).float()
        action_onehots.append(a_u_onehot)
    action_onehot = torch.cat(action_onehots, dim=1)

    conditions = torch.cat([states, action_onehot], dim=1)
    targets = torch.cat([next_states, tensors["terminals"]], dim=1)
    
    tensordata = TensorDataset(conditions, targets, tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])

    # 轉換為DataLoader
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "next_state_mean": next_state_mean,
        "next_state_std": next_state_std,
        "condition_dim": conditions.shape[1],
        "target_dim": targets.shape[1]
    }

    return dataloader, stats


# def prep_diffusion_dataloader(state_dim, dataset, num_UAVs, batch_size=256):
#     # 1. 拆分原始資料
#     states_df = dataset.iloc[:, 1 : state_dim + 1]
#     actions_df = dataset.iloc[:, state_dim + 1 : state_dim + num_UAVs + 1]
#     rewards_df = dataset.iloc[:,  state_dim + num_UAVs + 1]
#     next_states_df = dataset.iloc[:, state_dim + num_UAVs + 2 : state_dim + num_UAVs + state_dim + 2]
#     dones_df = dataset.iloc[:, state_dim + num_UAVs + state_dim + 2]

#     # 2. 轉成 tensor
#     states = torch.tensor(states_df.values, dtype=torch.float32)
#     actions = torch.tensor(actions_df.values, dtype=torch.long)
#     rewards = torch.tensor(rewards_df.values, dtype=torch.float32).unsqueeze(-1)
#     next_states = torch.tensor(next_states_df.values, dtype=torch.float32)
#     dones = torch.tensor(dones_df.values, dtype=torch.float32).unsqueeze(-1)

#     # 3. 對 state / next_state 做 normalization
#     state_mean = states.mean(dim=0, keepdim=True)
#     state_std = states.std(dim=0, keepdim=True).clamp(min=1e-6)
#     states = (states - state_mean) / state_std

#     next_state_mean = next_states.mean(dim=0, keepdim=True)
#     next_state_std = next_states.std(dim=0, keepdim=True).clamp(min=1e-6)
#     next_states = (next_states - next_state_mean) / next_state_std

#     # 4. 對每台 UAV 的 action 做 one-hot
#     action_onehots = []
#     for u in range(num_UAVs):
#         a_u = actions[:, u]
#         a_u_onehot = F.one_hot(a_u, num_classes=5 * (config.M + 1)).float()
#         action_onehots.append(a_u_onehot)

#     action_onehot = torch.cat(action_onehots, dim=1)

#     # 5. 組出 diffusion 用的 condition 與 target
#     conditions = torch.cat([states, action_onehot], dim=1)
#     targets = torch.cat([next_states, dones], dim=1)

#     # 6. 做成 DataLoader
#     tensordata = TensorDataset(conditions, targets)
#     dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

#     # 7. 把 normalization 資訊留著，之後生成完要還原會用到
#     stats = {
#         "state_mean": state_mean,
#         "state_std": state_std,
#         "next_state_mean": next_state_mean,
#         "next_state_std": next_state_std,
#         "condition_dim": conditions.shape[1],
#         "target_dim": targets.shape[1]
#     }

#     return dataloader, stats

class ConditionalDiffusionMLP(nn.Module):
    def __init__(self, target_dim, condition_dim, hidden_dim=256, time_embed_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        input_dim = target_dim + condition_dim + time_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x_t, t, condition):
        # x_t: (B, target_dim)
        # t:   (B,)
        # condition: (B, condition_dim)

        t = t.float().unsqueeze(1) / 1000.0  # (B,1)
        t_embed = self.time_mlp(t)  # (B,time_embed_dim)

        x = torch.cat([x_t, condition, t_embed], dim=1)
        pred_noise = self.net(x)
        return pred_noise

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def q_sample(x0, t, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t].unsqueeze(1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise

import torch
import torch.nn.functional as F

def train_diffusion(model, dataloader, optimizer, alpha_bars, T, device, epochs=20):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0

        for conditions, targets, _, _, _, _, _ in dataloader:
            conditions = conditions.to(device)
            targets = targets.to(device)

            B = targets.size(0)
            t = torch.randint(0, T, (B,), device=device)


            x_t, noise = q_sample(targets, t, alpha_bars)
            pred_noise = model(x_t, t, conditions)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B

        avg_loss = total_loss / total_count
        print(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")

@torch.no_grad()
def sample_diffusion(model, conditions, alphas, alpha_bars, betas, T, target_dim, device):
    model.eval()

    B = conditions.size(0)

    # 從純噪音開始
    x = torch.randn(B, target_dim, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        pred_noise = model(x, t_tensor, conditions)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        # DDPM 反向公式
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        if t > 0:
            z = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * z

    return x

def decode_action_onehot(action_onehot, num_UAVs, action_dim_per_uav):
    actions = []
    for u in range(num_UAVs):
        start = u * action_dim_per_uav
        end = (u + 1) * action_dim_per_uav
        a_u = action_onehot[:, start:end].argmax(dim=1, keepdim=True)
        actions.append(a_u)
    return torch.cat(actions, dim=1)

if __name__ == '__main__':

    df_src = fr"{config.PATH}\Datasets\Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen.csv"
    dataset = pd.read_csv(df_src)

    dataloader, stats = prep_dataloader(config.U*2 + config.M, dataset, config.U, batch_size=config.batch_size_offline)

    # print("conditions shape:", next(iter(dataloader))[0].shape)
    # print("targets shape:", next(iter(dataloader))[1].shape)
    # print("condition_dim:", stats["condition_dim"])
    # print("target_dim:", stats["target_dim"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalDiffusionMLP(
        target_dim=stats["target_dim"],
        condition_dim=stats["condition_dim"],
        hidden_dim=256,
        time_embed_dim=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    T = 1000
    betas, alphas, alpha_bars = make_beta_schedule(T, device=device)

    train_diffusion(model, dataloader, optimizer, alpha_bars, T, device, epochs=20)

    df = []

    for i in tqdm(range(400)):

        conditions, targets, real_s, real_a, real_reward, _, _ = next(iter(dataloader))
        conditions = conditions.to(device)
        targets = targets.to(device)

        fake_targets = sample_diffusion(
            model=model,
            conditions=conditions,
            alphas=alphas,
            alpha_bars=alpha_bars,
            betas=betas,
            T=T,
            target_dim=stats["target_dim"],
            device=device
        )

        fake_next_states_norm = fake_targets[:, :-1]
        fake_dones_raw = fake_targets[:, -1:]

        fake_next_states = fake_next_states_norm * stats["next_state_std"].to(device) + stats["next_state_mean"].to(device)
        fake_dones = (fake_dones_raw > 0.5).float()

        state_dim = config.U * 2 + config.M
        action_dim_total = stats["condition_dim"] - state_dim

        states_norm = conditions[:, :state_dim]
        action_onehot = conditions[:, state_dim:]

        states = states_norm * stats["state_std"].to(device) + stats["state_mean"].to(device)

        action_dim_per_uav = 5 * (config.M + 1)

        fake_actions = decode_action_onehot(
            action_onehot,
            num_UAVs=config.U,
            action_dim_per_uav=5 * (config.M + 1)
        )

        state_dim = config.U * 2 + config.M

        real_next_states_norm = targets[:, :state_dim]
        real_dones = targets[:, state_dim:]

        real_next_states = (
            real_next_states_norm * stats["next_state_std"].to(targets.device)
            + stats["next_state_mean"].to(targets.device)
        )

        fake_next_states = fake_next_states_norm * stats["next_state_std"].to(device) + stats["next_state_mean"].to(device)

        real_states_norm = conditions[:, :state_dim]

        real_states = (
            real_states_norm * stats["next_state_std"].to(targets.device)
            + stats["next_state_mean"].to(targets.device)
        )

        real_actions_norm = conditions[:, state_dim :]
        real_actions = decode_action_onehot(
            real_actions_norm,
            num_UAVs=config.U,
            action_dim_per_uav=5 * (config.M + 1)
        )




        # print("real next state[0]:", real_next_states[0])
        # print("fake next state[0]:", torch.round(fake_next_states[0]).clamp(min = 0.).float())

        # print("state", torch.round(real_states)[0].clamp(min = 0.).float())
        # print("real done[0]:", targets[0, -1])
        # print("fake done bin[0]:", fake_dones[0])

        states = real_s.cpu().numpy()
        actions = real_a.cpu().numpy()
        rewards = real_reward.cpu().numpy()
        fake_next_states = torch.round(fake_next_states[0]).clamp(min=0.).float().cpu().numpy()
        fake_done = targets[0, -1].cpu().numpy()

        # print(states[0].shape)
        # print(actions[0].shape)
        # print(rewards[0].shape)
        # print(fake_next_states.shape)
        # print(fake_done.shape)
        ds = np.concatenate((states[0], actions[0], rewards[0], fake_next_states, [fake_done]))
        # print(dataset.shape)
        if(i == 0):
            df = pd.DataFrame([ds])
        else:
            df.loc[len(df)] = ds

    df_src = fr"{config.PATH}\Datasets\Dataset_Online_DQN_FAKE_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"
    df.to_csv(df_src)

    # 從假資料集(ds)取100, 200, 400加入到真資料集(dataset)
    
    # 取得真資料集的前100, 200, 400筆資料
    fake_100 = df.head(100)
    fake_200 = df.head(200)
    fake_400 = df.head(400)
    
    # 合併真實資料集與假資料集
    dataset_100 = pd.concat([dataset, fake_100], ignore_index=True)
    dataset_200 = pd.concat([dataset, fake_200], ignore_index=True)
    dataset_400 = pd.concat([dataset, fake_400], ignore_index=True)
    
    # 儲存為CSV
    df_100 = fr"{config.PATH}\Datasets\Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_A100.csv"
    df_200 = fr"{config.PATH}\Datasets\Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_A200.csv"
    df_400 = fr"{config.PATH}\Datasets\Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_A400.csv"
    
    dataset_100.to_csv(df_100, index=False)
    dataset_200.to_csv(df_200, index=False)
    dataset_400.to_csv(df_400, index=False)