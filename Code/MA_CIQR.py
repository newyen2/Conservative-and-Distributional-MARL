
import pandas as pd
import torch
from utils import get_config, eval_runs_dist, prep_dataloader, RNGManager
from Code.agent_CQR import CQRAgent
from Environment import Environment
import matplotlib.pyplot as plt

def personalized_fedavg_shared_layers(agents, beta=0.5):
    shared_layer_names = [
        "head_1.weight", "head_1.bias",
        "ff_1.weight", "ff_1.bias"
    ]

    with torch.no_grad():
        state_dicts = [agent.network.state_dict() for agent in agents]
        avg_state = {}

        for key in shared_layer_names:
            avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

        for agent in agents:
            local_state = agent.network.state_dict()
            for key in shared_layer_names:
                local_state[key] = (1 - beta) * local_state[key] + beta * avg_state[key]
            agent.network.load_state_dict(local_state)

def Train_MA_CIQR(model, device_coord, risky_region, alpha, eta):
    # 初始化RNG, 環境與超參數
    config = get_config()
    rng = RNGManager()
    env = Environment(device_coord, risky_region, config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 預處理Dataset
    df_src = fr"{config.PATH}\Datasets\Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen.csv"
    dataset = pd.read_csv(df_src)
    dataloader = prep_dataloader(config.U*2 + config.M, dataset, config.U, batch_size=config.batch_size_offline)

    # 初始化Agent
    agents = []
    for u in range(config.U):
        agent = CQRAgent(state_size = env.nObservation,
                     action_size = env.nAction,
                     alpha = alpha,
                     eta = eta,
                     device = device)
        agents.append(agent)

    eval_rewards = []

    for epoch in range(1, config.epochs + 1):
        # 初始化
        loss = [0] * config.U

        # 從dataset進行CQL訓練
        for _, experience in enumerate(dataloader):

            # 拆分經驗樣本
            states, actions, rewards, next_states, dones = experience

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            for u in range(config.U):
                loss[u] = agents[u].Learn_CQR_ind((states, actions[:,[u]], rewards, next_states, dones))

        # personalized_fedavg_shared_layers(agents, beta=0.7)

        # for u in range(config.U):
        #     agents[u].soft_update(agents[u].network, agents[u].target_network)

        # 定時進行測試評估
        if epoch % config.eval_periods == 0:
            eval_reward = eval_runs_dist(env, agents, rng)
            eval_rewards.append(eval_reward)

            print(f"Epoch: {epoch} | Eval_Reward: {eval_reward} | Q Loss: {loss}")
    
    # 儲存模型參數
    for u in range(config.U):
        model_src = fr"{config.PATH}\Saved_Models\Model_Offline_{model}_{str(config.data_size)}%_UAV_{str(u)}_{str(config.penalty)}pen.pth"
        torch.save(agents[u].network.state_dict(), model_src)

    # 產生圖表
    fig_src = fr"{config.PATH}\Results\Result_Offline_{model}_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen.png"
    fig_data_src = fr"{config.PATH}\Result_Datas\Result_Offline_{model}_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen.csv"

    window_size = 10
    eval_rewards_smooth = pd.Series(eval_rewards).rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards_smooth, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'MA_{model} Reward')
    plt.legend()
    plt.grid(True)

    plt.savefig(fig_src)
    pd.DataFrame(eval_rewards).to_csv(fig_data_src, index=False)