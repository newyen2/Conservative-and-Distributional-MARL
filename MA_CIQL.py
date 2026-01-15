
import numpy as np
import pandas as pd
import torch
from utils import get_config, eval_runs, prep_dataloader
import random
from agent_CQL import CQLAgent
from Environment import environment
import matplotlib.pyplot as plt


def Train_MA_CIQL(Model,Dev_Coord,Risky_region,alpha):
    config = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = environment(Dev_Coord,Risky_region,config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Episode_Reward = []
    gamma = 0.99


    agent = []

    Num_UAVs_str = str(config.U)
    penalty_str = str(config.penalty)
    data_size_perc_str = str(config.data_size_perc)

    dataset = pd.read_csv(config.PATH+r'Datasets\Dataset_Online_DQN_'+data_size_perc_str+'%_'+Num_UAVs_str+'UAVs_pen_'+penalty_str+'.csv')

    dataloader = prep_dataloader(config.U*2 + config.M, dataset, config.U, batch_size=config.Batch_offline)

    for u in range(config.U):
        agent_u = CQLAgent(state_size=env.observation_space.shape,
                     action_size=env.action_space.shape[0],
                     alpha=alpha,
                     device=device)
        agent.append(agent_u)


    batches = 0
    eval_reward = eval_runs(env, agent)

    for i in range(1, config.epochs+1):
        loss = [0] * config.U
        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            for u in range(config.U):            
                loss[u], cql_loss, bellmann_error = agent[u].learn_cql_ind((states, actions[:,[u]], rewards, next_states, dones))

        if i % config.eval_every == 0:
            eval_reward = eval_runs(env, agent)

            Episode_Reward.append(eval_reward)
            print("Epoch: {} | Reward: {} | Q Loss_: {}".format(i, eval_reward, loss,))

    for u in range(config.U):
        u_str = str(u)
        torch.save(agent[u].network.state_dict(), config.PATH+r'Saved_Models\\'+Model+'_offline_'+data_size_perc_str+'%_UAV_'+u_str+'_pen_'+penalty_str+'.pth')

    plt.figure(figsize=(12, 6))
    window_size = 10
    Episode_Reward_smooth = pd.Series(Episode_Reward).rolling(window=window_size).mean()
    plt.plot(Episode_Reward_smooth, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'MA_{Model} Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(config.PATH+r'Results\\'+Model+'_offline_'+data_size_perc_str+'_pen_'+penalty_str+'.png')
    pd.DataFrame(Episode_Reward).to_csv(config.PATH+r'Result_Datas\\'+Model+'_offline_'+data_size_perc_str+'_pen_'+penalty_str+'.csv', index=False)
    # 讀取 CSV 以還原 Episode_Reward
    # Episode_Reward_loaded = pd.read_csv(config.PATH+r'Results\\'+Model+'_offline_'+data_size_perc_str+'%_UAV_'+u_str+'_pen_'+penalty_str+'_rewards.csv').iloc[:, 0].tolist()