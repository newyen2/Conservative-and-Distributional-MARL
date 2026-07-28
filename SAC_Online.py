import numpy as np
import pandas as pd
import torch
from buffer import ReplayBuffer
from utils import collect_random, get_config, eval_runs, RNGManager
from agent_Online_SAC import SACAgent
from Environment import Environment
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm
import sys

def Train_SAC_Online(device_coord,Risky_region):
    # TODO(HRL): 此函式保留舊 Hybrid SAC 訓練流程；階層式時序與雙 replay buffer 將由新入口負責。
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 初始化RNG, 環境與超參數
    config = get_config()
    rng = RNGManager()
    env = Environment(device_coord,Risky_region,config)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

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
    collect_steps = 500
    collect_random(env=env, U=config.U, dataset=buffers, steps=collect_steps)

    df = []

    # 試算表
    episode_df = {}
    episode_df['episode'] = []
    episode_df['init_UAV_pos'] = []
    episode_df['init_device_pos'] = []
    episode_df['risk_region'] = []

    step_df = {}
    step_df['episode'] = []
    step_df['step'] = []
    step_df['UAV_pos'] = []
    step_df['device_pos'] = []
    step_df['device_AOI'] = []
    step_df['UAV_action_select'] = []
    step_df['UAV_target_point'] = []
    step_df['reward'] = []

    # 評估獎勵
    eval_rewards = [] 

    pbar_episode = tqdm(
        range(1, config.episodes + 1),
        desc="Episode",
        position=0,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout
    )

    for episode in pbar_episode:
        # 初始化
        states = env.reset()
        actions = [0] * config.U
        episode_reward = 0.0 # 本次Episode的全局獎勵
        agent_rewards = np.zeros(config.U) # 個別Agent的獎勵

        episode_df['episode'].append(episode)
        episode_df['init_UAV_pos'].append(env.UAVs_init_coord)
        episode_df['init_device_pos'].append(env.device_coord.tolist())
        episode_df['risk_region'].append(env.risky_region)

        pbar_step = tqdm(
            range(100),
            desc="Step",
            position=1,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )

        for _ in pbar_step:
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

            step_df['episode'].append(episode)
            step_df['step'].append(steps)
            step_df['UAV_pos'].append(states.copy()[:4])
            step_df['device_AOI'].append(states.copy()[4:14])
            step_df['device_pos'].append(env.device_coord.tolist())
            step_df['UAV_action_select'].append([i[0] for i in actions])
            step_df['UAV_target_point'].append([i[1].tolist() for i in actions])
            step_df['reward'].append(rewards)

            if env.DONE:
                break

        tqdm.write(f"Episode: {episode} | Reward: {episode_reward} | Reward_u{agent_rewards} | Steps: {steps}")

        # 定時進行測試評估
        if episode % config.eval_periods == 0:
            eval_reward = eval_runs(env, agents, rng)
            eval_rewards.append(eval_reward)

            tqdm.write(f"Eval_Reward: {eval_reward}")

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Total training wall time: {end_time - start_time:.2f} seconds")

    # # 產生CSV
    df_src = fr"{config.PATH}\Stored_Datas\dataset.csv"
    
    save_start = int((config.episodes * env.max_steps)/2)
    save_end = int(save_start + config.data_size * (config.episodes * env.max_steps)/100)
    
    df = df.iloc[save_start:save_end]
    
    df.to_csv(df_src)
    
    # # 產生圖表
    fig_src = fr"{config.PATH}\Stored_Datas\result.png"
    fig_data_src = fr"{config.PATH}\Stored_Datas\result_data.csv"

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

    # 產生超參數資訊
    config_src = fr"{config.PATH}\Stored_Datas\config.json"

    configs = {}
    configs['general'] = vars(config)
    configs['agent'] = agents[0].get_parameter()
    configs['train'] = {
        'collect_steps': collect_steps,
        'df_src': df_src,
        'fig_src': fig_src,
        'fig_data_src': fig_data_src,
    }
    configs['info'] = {
        'date': datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"),
        'wall_time_second': round(end_time - start_time, 2)
    }

    pd.Series(configs).to_json(config_src, orient="index", indent=4)

    episode_df = pd.DataFrame(episode_df)
    episode_src = fr"{config.PATH}\Stored_Datas\episode.parquet"
    episode_df.to_parquet(episode_src, engine="pyarrow")

    step_df = pd.DataFrame(step_df)
    step_src = fr"{config.PATH}\Stored_Datas\step.parquet"
    step_df.to_parquet(step_src, engine="pyarrow")


def select_high_goals(agents, states, epsilon, deterministic=False):
    """新增理由：集中為每台 UAV 選擇一次高層 goal，之後由 Trainer 固定沿用 n 個環境步。"""
    pass


def collect_hierarchical_step(env, agents, states, goals, deterministic=False):
    """新增理由：固定執行 move、產生 S'、選 service、提交環境的順序，避免訓練與評估時序不同。"""
    pass


def Train_Hierarchical_Online(device_coord, Risky_region):
    """新增理由：以獨立入口管理高低層更新頻率及兩種 buffer，避免改壞仍可參考的舊 Trainer。"""
    pass
