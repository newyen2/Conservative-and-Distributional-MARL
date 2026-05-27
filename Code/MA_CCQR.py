import pandas as pd
import torch
from utils import get_config, eval_runs_dist, prep_dataloader, loss_update_cent, RNGManager
from Code.agent_CQR import CQRAgent
from Environment import Environment
import matplotlib.pyplot as plt

def Train_MA_CCQR(model,device_coord,risky_region,alpha,eta):
    # 初始化RNG, 環境與超參數
    config = get_config()
    rng = RNGManager()
    env = Environment(device_coord,risky_region,config)

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
        # 從dataset進行CQL訓練
        for _, experience in enumerate(dataloader):

             # 拆分經驗樣本
            states, actions, rewards, next_states, dones = experience

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # 價值聚合
            loss_CQL_c= 0
            Qt_expected_c = 0
            Qt_target_c = 0

            for u in range(config.U):            
                loss_CQL, Q_expected, Q_target = agents[u].Learn_CQR_cent((states, actions[:,[u]], rewards, next_states, dones))

                loss_CQL_c += loss_CQL
                Qt_expected_c += Q_expected
                Qt_target_c += Q_target

            # 計算Loss
            loss_tensor = agents[u].Loss_Calc(loss_CQL_c, Qt_expected_c, Qt_target_c)
    
            # 更新預測網路與目標網路
            network_param = list(agents[0].network.parameters()) + list(agents[1].network.parameters())
            target_param = list(agents[0].target_network.parameters()) + list(agents[1].target_network.parameters())
            optimizer = torch.optim.Adam(params = network_param, lr=1e-4)
            loss = loss_update_cent(loss_tensor, network_param ,target_param, optimizer)

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