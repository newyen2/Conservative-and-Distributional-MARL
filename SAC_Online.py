import numpy as np
import pandas as pd
import torch
from buffer import ReplayBuffer
from utils import collect_random, get_config, eval_runs, RNGManager
from agent_Online_SAC import SACAgent
from Environment import Environment
import matplotlib.pyplot as plt

def Train_SAC_Online(device_coord,Risky_region):
# 初始化RNG, 環境與超參數
    config = get_config()
    rng = RNGManager()
    env = Environment(device_coord,Risky_region,config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0

    # 初始化 Agent 與 ReplayBuffer
    agents = []
    buffers = []

    for i in range(config.U):
        agent = SACAgent(
            state_size=env.nObservation,
            continuous_action_size=env.nAction_move,
            discrete_action_size=env.nAction_select,
            device=device
        )

        buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.batch_size_online,
            device=device
        )

        agents.append(agent)
        buffers.append(buffer)

    # 透過隨機採取動作，於環境進行500步後，作為每個Agent的Replay Buffer
    collect_random(env=env, U=config.U, dataset=buffers, steps=500)

    # 試算表
    df = []

    # 評估獎勵
    eval_rewards = [] 


    for episode in range(1, config.episodes + 1):
        # 初始化
        states = env.reset()
        actions = [0] * config.U
        episode_reward = 0.0 # 本次Episode的全局獎勵
        agent_rewards = np.zeros(config.U) # 個別Agent的獎勵

        while True:
            # 個別UAV取得動作並執行
            for u in range(config.U):
                actions[u] = agents[u].get_action(states)

            next_states, rewards, done = env.step(states, actions)

            # 紀錄Experience至ReplayBuffer並採樣訓練
            for u in range(config.U):
                select_action = actions[u][0]
                continuous_action = np.asarray(actions[u][1], dtype=np.float32)

                action_vec = np.concatenate(
                    ([select_action], continuous_action),
                    axis=None
                ).astype(np.float32)

                buffers[u].add(states, action_vec, rewards[u], next_states, done[u])
                agent_rewards[u] += rewards[u]

                if env.done[u] == 0:
                    agents[u].Learn_SAC(buffers[u].sample())

            episode_reward += env.total_reward

            # 將Experiment紀錄為Offline Dataset
            flat_actions = []

            for u in range(config.U):
                select_action = actions[u][0]
                continuous_action = np.asarray(actions[u][1], dtype=np.float32)

                flat_actions.extend([select_action] + continuous_action.tolist())

            dataset = np.concatenate((
                    states,
                    np.asarray(flat_actions, dtype=np.float32),
                    [rewards[0]],
                    next_states,
                    [done[0]]
                ), axis=None)
            if(steps == 0):
                df = pd.DataFrame([dataset])
            else:
                df.loc[len(df)] = dataset

            steps += 1
            states = next_states.copy()

            if env.DONE:
                break

        print(f"Episode: {episode} | Reward: {episode_reward} | Reward_u{agent_rewards} | Steps: {steps}")

        # 定時進行測試評估
        if episode % config.eval_periods == 0:
            eval_reward = eval_runs(env, agents, rng)
            eval_rewards.append(eval_reward)

            print(f"Eval_Reward: {eval_reward}")
    

    # 產生CSV
    df_src = fr"{config.PATH}\Datasets\Dataset_Online_SAC_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"
    
    save_start = int((config.episodes * env.max_steps)/2)
    save_end = int(save_start + config.data_size * (config.episodes * env.max_steps)/100)
    
    df = df.iloc[save_start:save_end]
    
    df.to_csv(df_src)
    
    # 產生圖表
    fig_src = fr"{config.PATH}\Results\Result_Online_SAC_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.png"
    fig_data_src = fr"{config.PATH}\Result_Datas\Result_Online_SAC_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"

    window_size = 10
    eval_rewards_smooth = pd.Series(eval_rewards).rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards_smooth, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Online SAC Reward')
    plt.legend()
    plt.grid(True)

    plt.savefig(fig_src)
    pd.DataFrame(eval_rewards).to_csv(fig_data_src, index=False)