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


def Train_DSAC_T_Online(device_coord, Risky_region):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    config = get_config()
    rng = RNGManager()
    env = Environment(device_coord, Risky_region, config)

    # Keep the same default as your original file for comparability.
    # You can change this to cuda after confirming everything runs correctly.
    device = "cpu"

    steps = 0

    agents = []
    buffers = []

    for _ in range(config.U):
        agent = SACAgent(
            state_size=env.nObservation,
            continuous_action_size=env.nAction_move,
            discrete_action_size=env.nAction_select,
            device=device,
        )

        buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.batch_size_online,
            device=device,
        )

        agents.append(agent)
        buffers.append(buffer)

    collect_steps = 500
    collect_random(env=env, U=config.U, dataset=buffers, steps=collect_steps)

    df = []

    episode_df = {
        "episode": [],
        "init_UAV_pos": [],
        "init_device_pos": [],
        "risk_region": [],
    }

    step_df = {
        "episode": [],
        "step": [],
        "UAV_pos": [],
        "device_pos": [],
        "device_AOI": [],
        "UAV_action_select": [],
        "UAV_action_move": [],
        "reward": [],
    }

    eval_rewards = []
    train_logs = []

    pbar_episode = tqdm(
        range(1, config.episodes + 1),
        desc="Episode",
        position=0,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for episode in pbar_episode:
        states = env.reset()
        actions = [0] * config.U
        episode_reward = 0.0
        agent_rewards = np.zeros(config.U)

        episode_df["episode"].append(episode)
        episode_df["init_UAV_pos"].append(env.UAVs_init_coord)
        episode_df["init_device_pos"].append(env.device_coord.tolist())
        episode_df["risk_region"].append(env.risky_region)

        pbar_step = tqdm(
            range(env.max_steps),
            desc="Step",
            position=1,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for _ in pbar_step:
            for u in range(config.U):
                actions[u] = agents[u].get_action(states)

            next_states, rewards, done = env.step(states, actions)

            step_train_logs = []

            for u in range(config.U):
                select_action = actions[u][0]
                continuous_action = np.asarray(actions[u][1], dtype=np.float32)

                action_vec = np.concatenate(([select_action], continuous_action), axis=None).astype(np.float32)

                buffers[u].add(states, action_vec, rewards[u], next_states, done[u])
                agent_rewards[u] += rewards[u]

                if env.done[u] == 0:
                    loss_info = agents[u].Learn_DSAC_T(buffers[u].sample())
                    loss_info["episode"] = episode
                    loss_info["step"] = steps
                    loss_info["agent"] = u
                    step_train_logs.append(loss_info)

            train_logs.extend(step_train_logs)
            episode_reward += env.total_reward

            flat_actions = []
            for u in range(config.U):
                select_action = actions[u][0]
                continuous_action = np.asarray(actions[u][1], dtype=np.float32)
                flat_actions.extend([select_action] + continuous_action.tolist())

            dataset = np.concatenate(
                (
                    states,
                    np.asarray(flat_actions, dtype=np.float32),
                    [rewards[0]],
                    next_states,
                    [done[0]],
                ),
                axis=None,
            )

            if steps == 0:
                df = pd.DataFrame([dataset])
            else:
                df.loc[len(df)] = dataset

            steps += 1
            states = next_states.copy()

            step_df["episode"].append(episode)
            step_df["step"].append(steps)
            step_df["UAV_pos"].append(states.copy()[: config.U * 2])
            step_df["device_AOI"].append(states.copy()[config.U * 2: config.U * 2 + config.M])
            step_df["device_pos"].append(env.device_coord.tolist())
            step_df["UAV_action_select"].append([i[0] for i in actions])
            step_df["UAV_action_move"].append([i[1].tolist() for i in actions])
            step_df["reward"].append(rewards)

            if env.DONE:
                break

        tqdm.write(
            f"Episode: {episode} | Reward: {episode_reward} | "
            f"Reward_u{agent_rewards} | Steps: {steps}"
        )

        if episode % config.eval_periods == 0:
            eval_reward = eval_runs(env, agents, rng)
            eval_rewards.append(eval_reward)
            tqdm.write(f"Eval_Reward: {eval_reward}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Total training wall time: {end_time - start_time:.2f} seconds")

    df_src = fr"{config.PATH}\Datasets\Dataset_Online_DSAC_T_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"

    save_start = int((config.episodes * env.max_steps) / 2)
    save_end = int(save_start + config.data_size * (config.episodes * env.max_steps) / 100)

    df = df.iloc[save_start:save_end]
    df.to_csv(df_src)

    fig_src = fr"{config.PATH}\Results\Result_Online_DSAC_T_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.png"
    fig_data_src = fr"{config.PATH}\Result_Datas\Result_Online_DSAC_T_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"
    train_log_src = fr"{config.PATH}\Result_Datas\Train_Log_Online_DSAC_T_{str(config.data_size)}%_{str(config.U)}UAVs_{str(config.penalty)}pen_RND.csv"

    window_size = 10
    eval_rewards_smooth = pd.Series(eval_rewards).rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards_smooth, label="Reward")
    plt.xlabel("Evaluation Index")
    plt.ylabel("Reward")
    plt.title("Online Hybrid DSAC-T Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_src)
    plt.close()

    pd.DataFrame(eval_rewards).to_csv(fig_data_src, index=False)
    if len(train_logs) > 0:
        pd.DataFrame(train_logs).to_csv(train_log_src, index=False)

    config_src = fr"{config.PATH}\DSAC_T_Online_config.json"

    configs = {
        "general": vars(config),
        "agent": agents[0].get_parameter(),
        "train": {
            "collect_steps": collect_steps,
            "df_src": df_src,
            "fig_src": fig_src,
            "fig_data_src": fig_data_src,
            "train_log_src": train_log_src,
        },
        "info": {
            "date": datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"),
            "wall_time_second": round(end_time - start_time, 2),
        },
    }

    pd.Series(configs).to_json(config_src, orient="index", indent=4)

    episode_df = pd.DataFrame(episode_df)
    episode_src = fr"{config.PATH}\episode_DSAC_T.parquet"
    episode_df.to_parquet(episode_src, engine="pyarrow")

    step_df = pd.DataFrame(step_df)
    step_src = fr"{config.PATH}\step_DSAC_T.parquet"
    step_df.to_parquet(step_src, engine="pyarrow")

    return {
        "df_src": df_src,
        "fig_src": fig_src,
        "fig_data_src": fig_data_src,
        "train_log_src": train_log_src,
        "config_src": config_src,
        "episode_src": episode_src,
        "step_src": step_src,
    }
