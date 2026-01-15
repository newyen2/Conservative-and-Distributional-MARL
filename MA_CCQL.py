
import numpy as np
import pandas as pd
import torch
from utils import get_config, eval_runs, prep_dataloader, loss_update_cent
import random
from agent_CQL import CQLAgent
from Environment import environment
import matplotlib.pyplot as plt


def Train_MA_CCQL(Model,Dev_Coord,Risky_region,alpha):
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
        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            for u in range(config.U):            
                loss_CQL, Q_exp, Q_tar = agent[u].learn_cql_cent((states, actions[:,[u]], rewards, next_states, dones))

                if(u==0):
                    loss_CQL_All = loss_CQL
                    Q_exp_All = Q_exp
                    Q_tar_All = Q_tar
                else:
                    loss_CQL_All = loss_CQL_All + loss_CQL
                    Q_exp_All = Q_exp_All + Q_exp
                    Q_tar_All = Q_tar_All + Q_tar

            loss_tensor = agent[u].loss_calc_cent(loss_CQL_All,Q_exp_All,Q_tar_All)
            Net_params = list(agent[0].network.parameters()) + list(agent[1].network.parameters())
            Tar_params = list(agent[0].target_net.parameters()) + list(agent[1].target_net.parameters())

            optimizer = torch.optim.Adam(params=Net_params, lr=1e-4)
            loss = loss_update_cent(loss_tensor,Net_params,Tar_params,optimizer)


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

    
