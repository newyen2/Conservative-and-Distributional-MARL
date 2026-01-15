
import numpy as np
import pandas as pd
import torch
from buffer import ReplayBuffer
from utils import collect_random, get_config, eval_runs
import random
from agent_Online_DQN import DQNAgent
from Environment import environment
import matplotlib.pyplot as plt


def Train_DQN_Online(Dev_Coord,Risky_region):
    config = get_config()

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = environment(Dev_Coord,Risky_region,config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eps = 1.
    d_eps = 1 - config.min_eps
    steps = 0
    total_steps = 0

    agent = []
    buffer = []

    for i in range(config.U):
        agent_u = DQNAgent(state_size=env.observation_space.shape,
                     action_size=env.action_space.shape[0],
                     device=device)
        buffer_u = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.Batch_online, device=device)
        agent.append(agent_u)
        buffer.append(buffer_u)

    collect_random(env=env, U=config.U, dataset=buffer, num_samples=500)

    df_0 = pd.DataFrame()
    df_1 = pd.DataFrame()
    cntt = 0
    df = []

    Episode_Reward = []
    Episode_0_reward = []
    Episode_1_reward = []

    Eval_Reward = []


    for i in range(1, config.episodes+1):
        state = env.reset()
        episode_steps = 0
        rewards = 0
        action = [0] * config.U
        loss = [0] * config.U
        reward_all = np.zeros(config.U)

        while True:
            dataset_offline = []
            for u in range(config.U):
                action_u = agent[u].get_action(state, epsilon=eps)
                action[u] = action_u[0]
            steps += 1
            next_state, reward, done = env.step(state,action)

            for u in range(config.U):
                buffer[u].add(state, [action[u]], reward[u], next_state, done[u])
                reward_all[u] = reward_all[u] + reward[u]


            for u in range(config.U):
                if(env.done[u]==0):
                    loss[u], bellmann_error_0 = agent[u].learn_dqn(buffer[u].sample())

                    dataset_offline = np.concatenate((state,[action[u]],[reward[u]],next_state,[done[u]]))

                    if(cntt == 0):
                        df.append(pd.DataFrame([dataset_offline]))
                    else:
                        df[u].loc[len(df[u])] = dataset_offline


            dataset_offline_ALL_Agents = np.concatenate((state,action,[reward[0]],next_state,[done[0]]))
            if(cntt == 0):
                df_ALL_AGENTS=pd.DataFrame([dataset_offline_ALL_Agents])
            else:
                df_ALL_AGENTS.loc[len(df_ALL_AGENTS)] = dataset_offline_ALL_Agents

            state = next_state.copy()
            cntt = cntt + 1

            rewards += env.Total_reward
            episode_steps += 1
            eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
            if env.DONE:
                break

        Episode_Reward.append(rewards)
        Episode_0_reward.append(reward_all[0])
        Episode_1_reward.append(reward_all[1])

        total_steps += episode_steps
        print("Episode: {} | Reward: {} | reward_u{} | Steps: {}".format(i, rewards, reward_all, steps,))

        if i % config.eval_every == 0:
            eval_reward = eval_runs(env, agent)
            Eval_Reward.append(eval_reward)

            print("Episode: {} | Reward: {}".format(i, eval_reward))
    
    Num_UAVs_str = str(config.U)
    penalty_str = str(config.penalty)
    data_size_perc_str = str(config.data_size_perc)
    
    save_start = int((config.episodes * env.length_episode)/2)
    save_end = int(save_start + config.data_size_perc * (config.episodes * env.length_episode)/100)
    
    df_CTDE = df_ALL_AGENTS.iloc[save_start:save_end]
    df_CTDE.to_csv(config.PATH+r'Datasets\Dataset_Online_DQN_'+data_size_perc_str+'%_'+Num_UAVs_str+'UAVs_pen_'+penalty_str+'.csv')
    
    plt.figure(figsize=(12, 6))
    window_size = 10
    Episode_Reward_smooth = pd.Series(Eval_Reward).rolling(window=window_size).mean()
    plt.plot(Episode_Reward_smooth, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Online DQN Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(config.PATH+r'Results\\Result_Online_DQN_'+data_size_perc_str+'%_'+Num_UAVs_str+'UAVs_pen_'+penalty_str+'.png')
    pd.DataFrame(Eval_Reward).to_csv(config.PATH+r'Result_Datas\\Result_Online_DQN_'+data_size_perc_str+'_pen_'+penalty_str+'.csv', index=False)