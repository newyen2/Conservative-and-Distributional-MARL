"""階層式 SAC 的 Online Trainer。

本文件負責完整訓練生命週期：
1. 建立 Environment、每架 UAV 的 Agent 與高低層 ReplayBuffer。
2. 每 n 步更新一次 high-level goal。
3. 固定執行 move -> intermediate state -> service -> next state。
4. 每步保存 low-level transition，區段結束時保存 high-level transition。
5. 管理 warm-up、參數更新、評估、紀錄與存檔。

目前依照 MVP 流程逐步實作，尚未完成 ReplayBuffer 與完整訓練迴圈。
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from Environment import Environment
from agent_Onliine_HSAC import HSAC_Agent
from buffer import HighLevelReplayBuffer, LowLevelReplayBuffer
from utils import (
    RNGManager,
    get_config,
    linear_epsilon,
    sample_random_move,
)


class HSACOnlineTrainer:
    """管理多 UAV 階層式 RL 的完整 online training 流程。"""

    def __init__(self, config=None, device=None):
        """功能：建立 Trainer 設定、排程與狀態容器，實體元件由後續 build 方法建立。"""

        # [1. Config] 支援 argparse Namespace 或 dict，缺少的階層式參數使用 MVP 預設值。
        self.config = get_config() if config is None else config

        def config_value(name, default):
            if isinstance(self.config, dict):
                return self.config.get(name, default)
            return getattr(self.config, name, default)

        # [2. Device] 未指定時優先使用 CUDA；明確要求 CUDA 但不可用時直接回報。
        selected_device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(selected_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "指定使用 CUDA，但目前的 PyTorch 無法使用 CUDA。"
            )

        # [3. 基本環境與訓練規格]
        self.num_devices = max(
            1,
            int(config_value("M", 10)),
        )
        self.num_uavs = max(
            1,
            int(config_value("U", 2)),
        )
        self.num_episodes = max(
            1,
            int(config_value("episodes", 200)),
        )
        self.max_steps = max(
            1,
            int(config_value("max_steps", 100)),
        )
        self.max_movement = max(
            0.0,
            float(config_value("max_mov", 2.0)),
        )
        self.aoi_max = max(
            1,
            int(config_value("aoi_max", 100)),
        )
        self.energy_weight = float(
            config_value("energy_weight", 500.0)
        )
        self.risk_penalty = float(
            config_value(
                "risk_penalty",
                config_value("penalty", 300.0),
            )
        )
        self.risk_probability = min(
            1.0,
            max(
                0.0,
                float(
                    config_value(
                        "risk_probability",
                        config_value("prob_of_risk", 0.1),
                    )
                ),
            ),
        )
        # Base local reward 保留原 team reward 的平均值，同時讓每個 Critic
        # 直接對自己的 power 與 risk 負責；其餘 shaping 可獨立調整。
        self.goal_progress_weight = float(
            config_value("goal_progress_weight", 1.0)
        )
        self.wasted_move_weight = float(
            config_value("wasted_move_weight", 1.0)
        )
        self.goal_completion_bonus = float(
            config_value("goal_completion_bonus", 1.0)
        )
        self.high_interval = max(
            1,
            int(config_value("high_interval", 5)),
        )

        # [4. ReplayBuffer] High 與 low 可分別設定，未提供時沿用舊 SAC buffer 規格。
        default_buffer_size = int(
            config_value("buffer_size", 100_000)
        )
        default_batch_size = int(
            config_value("batch_size_online", 32)
        )
        self.low_buffer_size = max(
            1,
            int(
                config_value(
                    "low_buffer_size",
                    default_buffer_size,
                )
            ),
        )
        self.high_buffer_size = max(
            1,
            int(
                config_value(
                    "high_buffer_size",
                    default_buffer_size,
                )
            ),
        )
        self.low_batch_size = max(
            1,
            int(
                config_value(
                    "low_batch_size",
                    default_batch_size,
                )
            ),
        )
        self.high_batch_size = max(
            1,
            int(
                config_value(
                    "high_batch_size",
                    default_batch_size,
                )
            ),
        )
        self.warmup_steps = max(
            0,
            int(config_value("warmup_steps", 500)),
        )

        # [5. 更新排程] Low 通常每步更新，High 可使用獨立頻率。
        self.low_update_frequency = max(
            1,
            int(config_value("low_update_frequency", 1)),
        )
        self.high_update_frequency = max(
            1,
            int(
                config_value(
                    "high_update_frequency",
                    self.high_interval,
                )
            ),
        )
        self.updates_per_step = max(
            1,
            int(config_value("updates_per_step", 1)),
        )

        # [6. High DQN 探索] epsilon 由 Trainer 依 global_step 控制。
        self.epsilon_start = float(
            config_value("epsilon_start", 1.0)
        )
        self.epsilon_end = float(
            config_value(
                "epsilon_end",
                config_value("min_eps", 0.01),
            )
        )
        self.epsilon_decay_steps = max(
            1,
            int(
                config_value(
                    "epsilon_decay_steps",
                    config_value("eps_frames", 10_000),
                )
            ),
        )
        self.high_gamma = float(
            config_value("high_gamma", 0.99)
        )
        self.low_gamma = float(
            config_value("low_gamma", 0.99)
        )
        self.hidden_size = max(
            1,
            int(config_value("hidden_size", 256)),
        )
        self.high_target_update_interval = max(
            1,
            int(config_value("high_target_update_interval", 100)),
        )
        self.gradient_clip_norm = max(
            0.0,
            float(config_value("gradient_clip_norm", 1.0)),
        )
        self.show_progress = not bool(
            config_value("quiet", False)
        )
        self.progress_interval = max(
            1,
            int(config_value("progress_interval", 25)),
        )

        # [7. 評估與輸出]
        self.eval_period = max(
            1,
            int(
                config_value(
                    "eval_period",
                    config_value("eval_periods", 1),
                )
            ),
        )
        self.eval_episodes = max(
            1,
            int(config_value("eval_episodes", 10)),
        )
        configured_output_dir = config_value(
            "output_dir",
            None,
        )
        if configured_output_dir:
            output_path = Path(str(configured_output_dir))
        else:
            legacy_path = Path(
                str(config_value("PATH", "."))
            )
            output_path = (
                legacy_path
                if legacy_path.name.lower() == "stored_datas"
                else legacy_path / "Stored_Datas"
            )
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        self.output_path = str(output_path.resolve())

        # [8. RNG] 沿用 utils 的 train/eval 隨機狀態切換器。
        self.rng = RNGManager(config=self.config)

        # [9. 實體元件] 由 build_* 方法建立，避免 __init__ 隱含啟動環境或訓練。
        self.env = None
        self.state_size = None
        self.agents = []
        self.low_buffers = []
        self.high_buffers = []

        # [10. 全域訓練狀態]
        self.global_step = 0
        self.current_episode = 0
        self.current_state = None
        self.current_goals = None
        self.terminated = False

        # [11. High segment context] 每架 UAV 各自保存區段起點、回報與 duration。
        self.segment_start_states = None
        self.segment_goals = None
        self.segment_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        self.segment_durations = np.zeros(
            self.num_uavs,
            dtype=np.int32,
        )
        self.segment_goal_completed = np.zeros(
            self.num_uavs,
            dtype=np.bool_,
        )

        # [12. 訓練紀錄] initialize_logs() 後再建立正式欄位。
        self.episode_logs = []
        self.step_logs = []
        self.eval_logs = []
        self.loss_logs = []

        # [13. 執行時間與最佳模型資訊]
        self.training_start_time = None
        self.training_end_time = None
        self.best_eval_return = None
        self.best_checkpoint_path = None

    # ------------------------------------------------------------------
    # 建立訓練元件
    # ------------------------------------------------------------------
    def build_environment(self):
        """功能：依 config 建立 Environment，並取得 state、M、U 與動作維度。"""
        self.env = Environment(
            num_devices=self.num_devices,
            num_uavs=self.num_uavs,
            max_steps=self.max_steps,
            max_movement=self.max_movement,
            aoi_max=self.aoi_max,
            energy_weight=self.energy_weight,
            risk_penalty=self.risk_penalty,
            risk_probability=self.risk_probability,
        )
        self.env.reset()

        # 固定場景目前只建立 10 個裝置與 2 架 UAV，先在 Trainer 邊界明確檢查。
        if self.env.device_coord.shape[0] != self.num_devices:
            raise ValueError(
                "Environment.reset() 建立的裝置數量與 config.M 不一致："
                f"{self.env.device_coord.shape[0]} != {self.num_devices}。"
            )
        if self.env.uav_locations.shape[0] != self.num_uavs:
            raise ValueError(
                "Environment.reset() 建立的 UAV 數量與 config.U 不一致："
                f"{self.env.uav_locations.shape[0]} != {self.num_uavs}。"
            )

        self.current_state = self.env.get_state()
        self.state_size = int(self.current_state.shape[0])
        self.terminated = bool(self.env.terminated)

        return self.env

    def build_agents(self):
        """功能：為每架 UAV 建立一個 HSAC_Agent，並注入各自的 move preview function。"""
        if self.env is None:
            self.build_environment()

        self.agents = []
        for uav_index in range(self.num_uavs):
            agent = HSAC_Agent(
                state_size=self.state_size,
                num_devices=self.num_devices,
                num_uavs=self.num_uavs,
                move_action_size=2,
                high_interval=self.high_interval,
                max_movement=self.max_movement,
                hidden_size=self.hidden_size,
                device=self.device,
                high_target_update_interval=(
                    self.high_target_update_interval
                ),
                gradient_clip_norm=self.gradient_clip_norm,
                move_preview_fn=self.build_move_preview_fn(uav_index),
            )

            # 折扣率由 Trainer config 管理，再同步給各 Agent。
            agent.high_gamma = self.high_gamma
            agent.low_gamma = self.low_gamma
            self.agents.append(agent)

        return self.agents

    def build_replay_buffers(self):
        """功能：為每個 Agent 建立獨立的 high-level 與 low-level ReplayBuffer。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        self.low_buffers = [
            LowLevelReplayBuffer(
                buffer_size=self.low_buffer_size,
                batch_size=self.low_batch_size,
                device=self.device,
            )
            for _ in range(self.num_uavs)
        ]
        self.high_buffers = [
            HighLevelReplayBuffer(
                buffer_size=self.high_buffer_size,
                batch_size=self.high_batch_size,
                device=self.device,
            )
            for _ in range(self.num_uavs)
        ]

        return self.low_buffers, self.high_buffers

    def build_move_preview_fn(self, uav_index):
        """功能：建立無副作用的 batch move preview，供指定 UAV 計算 SAC target 中間態。"""
        if self.env is None:
            self.build_environment()

        if not 0 <= int(uav_index) < self.num_uavs:
            raise ValueError(
                f"uav_index 必須介於 0 與 {self.num_uavs - 1}。"
            )

        location_start = int(uav_index) * 2
        location_end = location_start + 2
        map_low = torch.as_tensor(
            self.env.map_low,
            dtype=torch.float32,
            device=self.device,
        )
        map_high = torch.as_tensor(
            self.env.map_high,
            dtype=torch.float32,
            device=self.device,
        )
        max_movement = torch.as_tensor(
            self.max_movement,
            dtype=torch.float32,
            device=self.device,
        )

        def preview_move(states, move_actions):
            """複製 state 並只更新指定 UAV 的座標，不改動真實環境。"""
            state_tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )
            action_tensor = torch.as_tensor(
                move_actions,
                dtype=torch.float32,
                device=self.device,
            )

            state_was_unbatched = state_tensor.ndim == 1
            if state_was_unbatched:
                state_tensor = state_tensor.unsqueeze(0)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)

            if state_tensor.shape[-1] != self.state_size:
                raise ValueError(
                    "move preview 的 state 維度不正確："
                    f"{state_tensor.shape[-1]} != {self.state_size}。"
                )
            if action_tensor.shape[-1] != 2:
                raise ValueError(
                    "move preview 的 action 最後一維必須是 2。"
                )

            # 單筆 action 可套用到整個 state batch；其餘情況需逐筆對齊。
            if action_tensor.shape[0] == 1 and state_tensor.shape[0] > 1:
                action_tensor = action_tensor.expand(
                    state_tensor.shape[0],
                    -1,
                )
            elif action_tensor.shape[0] != state_tensor.shape[0]:
                raise ValueError(
                    "move preview 的 state 與 action batch 大小不一致。"
                )

            # 與 Environment.Update_Location 相同：限制向量長度後再裁切地圖邊界。
            action_norm = torch.linalg.vector_norm(
                action_tensor,
                dim=-1,
                keepdim=True,
            )
            movement_scale = torch.clamp(
                max_movement / (action_norm + 1e-8),
                max=1.0,
            )
            bounded_action = action_tensor * movement_scale

            preview_states = state_tensor.clone()
            current_locations = preview_states[
                :,
                location_start:location_end,
            ]
            preview_states[:, location_start:location_end] = torch.clamp(
                current_locations + bounded_action,
                min=map_low,
                max=map_high,
            )

            if state_was_unbatched:
                return preview_states.squeeze(0)
            return preview_states

        return preview_move

    def initialize_logs(self):
        """功能：建立 episode、step、evaluation、loss 與執行時間的紀錄容器。"""
        self.episode_logs = []
        self.step_logs = []
        self.eval_logs = []
        self.loss_logs = []

        return {
            "episode_logs": self.episode_logs,
            "step_logs": self.step_logs,
            "eval_logs": self.eval_logs,
            "loss_logs": self.loss_logs,
        }

    # ------------------------------------------------------------------
    # Episode 與 high-level context
    # ------------------------------------------------------------------
    def reset_episode(self):
        """功能：重置 Environment，並初始化 goal、區段起點、累積回報與 duration。"""
        if self.env is None:
            self.build_environment()
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        self.env.reset()
        self.current_state = self.env.get_state()
        self.terminated = bool(self.env.terminated)

        # Goal 需依當前 epsilon 另外選擇；此處只清除上一個 episode 的決策。
        self.current_goals = None
        self.segment_goals = None
        self.segment_start_states = np.repeat(
            self.current_state[np.newaxis, :],
            self.num_uavs,
            axis=0,
        ).astype(np.float32, copy=False)
        self.segment_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        self.segment_durations = np.zeros(
            self.num_uavs,
            dtype=np.int32,
        )
        self.segment_goal_completed = np.zeros(
            self.num_uavs,
            dtype=np.bool_,
        )

        return self.current_state.copy()

    def should_refresh_goal(self, goal_duration, done):
        """功能：判斷 high goal 是否已持續 n 步，或因 episode 結束而關閉區段。"""
        durations = np.asarray(goal_duration)
        done_flags = np.asarray(done, dtype=np.bool_)
        refresh = np.logical_or(
            durations >= self.high_interval,
            done_flags,
        )

        if refresh.ndim == 0:
            return bool(refresh)
        return refresh.astype(np.bool_, copy=False)

    def select_high_goals(
        self,
        states,
        epsilon,
        deterministic=False,
    ):
        """功能：集中呼叫所有 Agent 的 High DQN，產生每架 UAV 的長期 goal。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        goals = np.empty(
            self.num_uavs,
            dtype=np.int64,
        )
        for agent_index, agent in enumerate(self.agents):
            agent_state = self._state_for_agent(
                states,
                agent_index,
                "states",
            )
            goals[agent_index] = agent.get_high_action(
                agent_state,
                epsilon=epsilon,
                deterministic=deterministic,
            )

        return goals

    def update_high_context(
        self,
        rewards,
        next_state,
        done,
        epsilon,
    ):
        """功能：累積區段折扣回報、增加 duration，並決定下一步沿用或更新 goal。"""
        if self.current_goals is None or self.segment_goals is None:
            raise RuntimeError(
                "更新 high context 前必須先選擇 current_goals。"
            )

        reward_values = np.asarray(
            [
                self._scalar_for_agent(
                    rewards,
                    agent_index,
                    "rewards",
                )
                for agent_index in range(self.num_uavs)
            ],
            dtype=np.float32,
        )

        # 區段回報使用 step-wise 折扣，與 high target 的 gamma**duration 對齊。
        discounts = np.power(
            self.high_gamma,
            self.segment_durations,
        ).astype(np.float32, copy=False)
        self.segment_returns += discounts * reward_values
        self.segment_durations += 1

        refresh_mask = self.finalize_high_transitions(
            next_state=next_state,
            done=done,
        )
        done_flags = np.asarray(
            [
                self._scalar_for_agent(
                    done,
                    agent_index,
                    "done",
                    allow_shared=True,
                )
                for agent_index in range(self.num_uavs)
            ],
            dtype=np.bool_,
        )
        self.terminated = bool(np.any(done_flags))

        # 結束的 Agent 不再建立新 goal；其餘到期 Agent從 next_state 選擇下一段目標。
        active_refresh = refresh_mask & ~done_flags
        for agent_index in np.flatnonzero(active_refresh):
            agent_state = self._state_for_agent(
                next_state,
                int(agent_index),
                "next_state",
            )
            new_goal = self.agents[int(agent_index)].get_high_action(
                agent_state,
                epsilon=epsilon,
                deterministic=False,
            )
            self.current_goals[int(agent_index)] = new_goal
            self.segment_goals[int(agent_index)] = new_goal
            self.segment_goal_completed[int(agent_index)] = False

        return self.current_goals.copy(), refresh_mask

    def finalize_high_transitions(
        self,
        next_state,
        done,
    ):
        """功能：將已完成 goal 區段寫入 high buffer，並建立下一個區段起點。"""
        if self.current_goals is None or self.segment_goals is None:
            raise RuntimeError(
                "完成 high transition 前必須先選擇 current_goals。"
            )
        if len(self.high_buffers) != self.num_uavs:
            self.build_replay_buffers()

        refresh_mask = np.asarray(
            self.should_refresh_goal(
                self.segment_durations,
                done,
            ),
            dtype=np.bool_,
        )
        if refresh_mask.ndim == 0:
            refresh_mask = np.full(
                self.num_uavs,
                bool(refresh_mask),
                dtype=np.bool_,
            )

        # duration=0 代表區段尚未真正執行，不建立空 transition。
        refresh_mask &= self.segment_durations > 0

        for agent_index in np.flatnonzero(refresh_mask):
            agent_index = int(agent_index)
            done_value = self._scalar_for_agent(
                done,
                agent_index,
                "done",
                allow_shared=True,
            )
            agent_next_state = self._state_for_agent(
                next_state,
                agent_index,
                "next_state",
            )
            self.high_buffers[agent_index].add(
                state=self.segment_start_states[agent_index],
                goal=self.segment_goals[agent_index],
                cumulative_reward=self.segment_returns[agent_index],
                next_state=agent_next_state,
                done=done_value,
                duration=self.segment_durations[agent_index],
            )

            # 新區段從當前 next_state 開始；goal 由 update_high_context 接著更新。
            self.segment_start_states[agent_index] = agent_next_state
            self.segment_returns[agent_index] = 0.0
            self.segment_durations[agent_index] = 0

        return refresh_mask

    # ------------------------------------------------------------------
    # 兩階段環境互動
    # ------------------------------------------------------------------
    def select_move_actions(
        self,
        state,
        goals,
        deterministic=False,
    ):
        """功能：取得所有 Agent 的二維 move action，組成 Environment 所需的 (U, 2)。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        move_actions = np.empty(
            (self.num_uavs, 2),
            dtype=np.float32,
        )
        for agent_index, agent in enumerate(self.agents):
            agent_state = self._state_for_agent(
                state,
                agent_index,
                "state",
            )
            agent_goal = self._scalar_for_agent(
                goals,
                agent_index,
                "goals",
            )
            move_actions[agent_index] = agent.get_move_action(
                agent_state,
                agent_goal,
                deterministic=deterministic,
            )

        return move_actions

    def build_service_action_masks(self, moved_state, goals):
        """功能：依服務規則建立每架 UAV 的 M+1 維合法 action mask。"""
        # 先驗證每架 UAV 都有可供 SelectActor 使用的 state 與 goal。
        for agent_index in range(self.num_uavs):
            self._state_for_agent(
                moved_state,
                agent_index,
                "moved_state",
            )
            self._scalar_for_agent(
                goals,
                agent_index,
                "goals",
            )

        # MVP 尚未定義距離門檻或服務衝突；M 個裝置與「不服務」目前皆合法。
        return np.ones(
            (self.num_uavs, self.num_devices + 1),
            dtype=np.bool_,
        )

    def select_service_actions(
        self,
        moved_state,
        goals,
        action_masks=None,
        deterministic=False,
    ):
        """功能：在 move_step 後取得所有 Agent 的 service action。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()
        if action_masks is None:
            action_masks = self.build_service_action_masks(
                moved_state,
                goals,
            )

        service_actions = np.empty(
            self.num_uavs,
            dtype=np.int64,
        )
        for agent_index, agent in enumerate(self.agents):
            agent_state = self._state_for_agent(
                moved_state,
                agent_index,
                "moved_state",
            )
            agent_goal = self._scalar_for_agent(
                goals,
                agent_index,
                "goals",
            )
            agent_mask = self._mask_for_agent(
                action_masks,
                agent_index,
            )
            service_actions[agent_index] = agent.get_service_action(
                agent_state,
                agent_goal,
                action_mask=agent_mask,
                deterministic=deterministic,
            )

        return service_actions

    def calculate_step_rewards(
        self,
        state,
        moved_state,
        next_state,
        move_info,
        service_info,
        move_actions=None,
        goals=None,
        goal_completed=None,
        return_info=False,
    ):
        """功能：計算可歸因的 local reward，並加入 goal 與無效移動 shaping。"""
        # 驗證三種 state 都維持相同的全域 observation 契約。
        for agent_index in range(self.num_uavs):
            self._state_for_agent(
                state,
                agent_index,
                "state",
            )
            self._state_for_agent(
                moved_state,
                agent_index,
                "moved_state",
            )
            self._state_for_agent(
                next_state,
                agent_index,
                "next_state",
            )

        power = np.asarray(
            service_info["power"],
            dtype=np.float32,
        ).reshape(-1)
        risk_triggered = np.asarray(
            move_info["risk_triggered"],
            dtype=np.float32,
        ).reshape(-1)

        if power.size != self.num_uavs:
            raise ValueError(
                "service_info['power'] 必須為每架 UAV 提供一個值。"
            )
        if risk_triggered.size != self.num_uavs:
            raise ValueError(
                "move_info['risk_triggered'] 必須為每架 UAV 提供一個值。"
            )

        # 每個 Agent 使用自己的 power/risk；base rewards 的平均仍等於
        # 舊公式的 team reward，額外 shaping 則只用於改善學習訊號。
        aoi_cost = float(service_info["mean_aoi"])
        base_rewards = -(
            aoi_cost
            + self.energy_weight * power
            + self.risk_penalty * risk_triggered
        )

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        moved_state_array = np.asarray(
            moved_state,
            dtype=np.float32,
        ).reshape(-1)
        current_locations = state_array[
            : self.num_uavs * 2
        ].reshape(self.num_uavs, 2)
        moved_locations = moved_state_array[
            : self.num_uavs * 2
        ].reshape(self.num_uavs, 2)
        device_start = self.num_uavs * 2 + self.num_devices
        device_coord = state_array[
            device_start:
            device_start + self.num_devices * 2
        ].reshape(self.num_devices, 2)

        goal_progress = np.zeros(self.num_uavs, dtype=np.float32)
        new_goal_completion = np.zeros(
            self.num_uavs,
            dtype=np.bool_,
        )
        goal_values = None
        if goals is not None:
            goal_values = np.asarray(
                goals,
                dtype=np.int64,
            ).reshape(self.num_uavs)
            valid_goals = (
                (goal_values >= 0)
                & (goal_values < self.num_devices)
            )
            for agent_index in np.flatnonzero(valid_goals):
                target = device_coord[goal_values[agent_index]]
                before_distance = np.linalg.norm(
                    current_locations[agent_index] - target
                )
                after_distance = np.linalg.norm(
                    moved_locations[agent_index] - target
                )
                goal_progress[agent_index] = (
                    before_distance - after_distance
                )

            completed_before = (
                np.zeros(self.num_uavs, dtype=np.bool_)
                if goal_completed is None
                else np.asarray(
                    goal_completed,
                    dtype=np.bool_,
                ).reshape(self.num_uavs)
            )
            service_targets = np.asarray(
                service_info["service_targets"],
                dtype=np.int64,
            ).reshape(self.num_uavs)
            new_goal_completion = (
                valid_goals
                & ~completed_before
                & (service_targets == goal_values)
            )

        executed_moves = np.asarray(
            move_info["movement"],
            dtype=np.float32,
        ).reshape(self.num_uavs, 2)
        if move_actions is None:
            command_moves = executed_moves
        else:
            command_moves = np.asarray(
                move_actions,
                dtype=np.float32,
            ).reshape(self.num_uavs, 2)
        wasted_move = np.linalg.norm(
            command_moves - executed_moves,
            axis=1,
        ).astype(np.float32, copy=False)

        movement_scale = max(self.max_movement, 1e-6)
        normalized_progress = np.clip(
            goal_progress / movement_scale,
            -1.0,
            1.0,
        )
        normalized_waste = np.clip(
            wasted_move / movement_scale,
            0.0,
            1.0,
        )
        rewards = (
            base_rewards
            + self.goal_progress_weight * normalized_progress
            - self.wasted_move_weight * normalized_waste
            + self.goal_completion_bonus
            * new_goal_completion.astype(np.float32)
        ).astype(np.float32, copy=False)

        reward_info = {
            "base_rewards": base_rewards.astype(
                np.float32,
                copy=False,
            ),
            "goal_progress": goal_progress,
            "normalized_goal_progress": normalized_progress.astype(
                np.float32,
                copy=False,
            ),
            "new_goal_completion": new_goal_completion,
            "wasted_move": wasted_move,
            "normalized_wasted_move": normalized_waste.astype(
                np.float32,
                copy=False,
            ),
        }
        if return_info:
            return rewards, reward_info
        return rewards

    def collect_environment_step(
        self,
        state,
        goals,
        deterministic=False,
    ):
        """功能：固定執行 move、服務與 reward 計算，回傳完整單步互動資料。"""
        if self.env is None:
            self.build_environment()

        move_actions = self.select_move_actions(
            state,
            goals,
            deterministic=deterministic,
        )
        moved_state, move_info = self.env.move_step(
            move_actions
        )

        action_masks = self.build_service_action_masks(
            moved_state,
            goals,
        )
        service_actions = self.select_service_actions(
            moved_state,
            goals,
            action_masks=action_masks,
            deterministic=deterministic,
        )
        next_state, done, service_info = self.env.service_step(
            service_actions
        )

        rewards, reward_info = self.calculate_step_rewards(
            state=state,
            moved_state=moved_state,
            next_state=next_state,
            move_info=move_info,
            service_info=service_info,
            move_actions=move_actions,
            goals=goals,
            goal_completed=self.segment_goal_completed,
            return_info=True,
        )
        self.segment_goal_completed |= reward_info[
            "new_goal_completion"
        ]

        self.current_state = next_state.copy()
        self.terminated = bool(done)

        return {
            "state": np.asarray(
                state,
                dtype=np.float32,
            ).copy(),
            "goals": np.asarray(
                goals,
                dtype=np.int64,
            ).copy(),
            "move_actions": move_actions,
            "moved_state": moved_state,
            "action_masks": action_masks,
            "service_actions": service_actions,
            "rewards": rewards,
            "next_state": next_state,
            "done": bool(done),
            "move_info": move_info,
            "service_info": service_info,
            "reward_info": reward_info,
        }

    # ------------------------------------------------------------------
    # ReplayBuffer 管理
    # ------------------------------------------------------------------
    def add_low_transitions(
        self,
        state,
        goals,
        move_actions,
        moved_state,
        service_actions,
        rewards,
        next_state,
        next_goals,
        done,
        action_masks=None,
        next_action_masks=None,
        executed_moves=None,
    ):
        """功能：每個 timestep 為所有 Agent 保存一筆 low-level transition。"""
        if len(self.low_buffers) != self.num_uavs:
            self.build_replay_buffers()

        for agent_index in range(self.num_uavs):
            self.low_buffers[agent_index].add(
                state=self._state_for_agent(
                    state,
                    agent_index,
                    "state",
                ),
                goal=self._scalar_for_agent(
                    goals,
                    agent_index,
                    "goals",
                ),
                move_action=self._vector_for_agent(
                    move_actions,
                    agent_index,
                    "move_actions",
                ),
                moved_state=self._state_for_agent(
                    moved_state,
                    agent_index,
                    "moved_state",
                ),
                service_action=self._scalar_for_agent(
                    service_actions,
                    agent_index,
                    "service_actions",
                ),
                reward=self._scalar_for_agent(
                    rewards,
                    agent_index,
                    "rewards",
                ),
                next_state=self._state_for_agent(
                    next_state,
                    agent_index,
                    "next_state",
                ),
                next_goal=self._scalar_for_agent(
                    next_goals,
                    agent_index,
                    "next_goals",
                ),
                done=self._scalar_for_agent(
                    done,
                    agent_index,
                    "done",
                    allow_shared=True,
                ),
                action_mask=self._mask_for_agent(
                    action_masks,
                    agent_index,
                ),
                next_action_mask=self._mask_for_agent(
                    next_action_masks,
                    agent_index,
                ),
                executed_move=self._vector_for_agent(
                    (
                        move_actions
                        if executed_moves is None
                        else executed_moves
                    ),
                    agent_index,
                    "executed_moves",
                ),
            )

        return self.num_uavs

    def add_high_transitions(
        self,
        segment_states,
        goals,
        cumulative_rewards,
        next_state,
        done,
        durations,
    ):
        """功能：goal 更新或 episode 結束時，保存各 Agent 的 high-level transition。"""
        if len(self.high_buffers) != self.num_uavs:
            self.build_replay_buffers()

        for agent_index in range(self.num_uavs):
            self.high_buffers[agent_index].add(
                state=self._state_for_agent(
                    segment_states,
                    agent_index,
                    "segment_states",
                ),
                goal=self._scalar_for_agent(
                    goals,
                    agent_index,
                    "goals",
                ),
                cumulative_reward=self._scalar_for_agent(
                    cumulative_rewards,
                    agent_index,
                    "cumulative_rewards",
                ),
                next_state=self._state_for_agent(
                    next_state,
                    agent_index,
                    "next_state",
                ),
                done=self._scalar_for_agent(
                    done,
                    agent_index,
                    "done",
                    allow_shared=True,
                ),
                duration=self._scalar_for_agent(
                    durations,
                    agent_index,
                    "durations",
                ),
            )

        return self.num_uavs

    def _state_for_agent(self, value, agent_index, name):
        """讀取共享 state，或從 (U, S) 狀態陣列取得指定 Agent 的 state。"""
        array = np.asarray(value)
        if array.ndim == 1:
            return array
        if array.ndim == 2 and array.shape[0] == self.num_uavs:
            return array[agent_index]
        raise ValueError(
            f"{name} 必須是 (S,) 或 (U, S)，目前 shape={array.shape}。"
        )

    def _scalar_for_agent(
        self,
        value,
        agent_index,
        name,
        allow_shared=False,
    ):
        """從 scalar 或每架 UAV 的陣列取得指定 Agent 的值。"""
        array = np.asarray(value)
        if array.ndim == 0:
            if allow_shared:
                return array.item()
            if self.num_uavs == 1:
                return array.item()
        if array.size == 1 and allow_shared:
            return array.reshape(-1)[0].item()
        if array.shape[0] == self.num_uavs:
            selected = np.asarray(array[agent_index])
            if selected.size == 1:
                return selected.item()
        raise ValueError(
            f"{name} 必須為每架 UAV 提供一個值，目前 shape={array.shape}。"
        )

    def _vector_for_agent(self, value, agent_index, name):
        """從 (U, action_dim) 陣列取得指定 Agent 的動作向量。"""
        array = np.asarray(value)
        if self.num_uavs == 1 and array.ndim == 1:
            return array
        if array.ndim == 2 and array.shape[0] == self.num_uavs:
            return array[agent_index]
        raise ValueError(
            f"{name} 必須是 (U, action_dim)，目前 shape={array.shape}。"
        )

    def _mask_for_agent(self, masks, agent_index):
        """讀取共享 mask，或從 (U, M+1) 陣列取得指定 Agent 的 mask。"""
        if masks is None:
            return None

        array = np.asarray(masks, dtype=np.bool_)
        if array.ndim == 1:
            return array
        if array.ndim == 2 and array.shape[0] == self.num_uavs:
            return array[agent_index]
        raise ValueError(
            "action mask 必須是 (M+1,) 或 (U, M+1)，"
            f"目前 shape={array.shape}。"
        )

    def buffers_ready(self, level, agent_index):
        """功能：確認指定 Agent 的 high 或 low buffer 是否已有足夠 batch 樣本。"""
        if level == "low":
            buffers = self.low_buffers
        elif level == "high":
            buffers = self.high_buffers
        else:
            raise ValueError("level 必須是 'low' 或 'high'。")

        if not 0 <= int(agent_index) < len(buffers):
            return False

        buffer = buffers[int(agent_index)]
        return len(buffer) >= buffer.batch_size

    def sample_agent_experiences(self, level, agent_index):
        """功能：從指定 Agent 的 high 或 low buffer 取出一個訓練 batch。"""
        if level == "low":
            buffers = self.low_buffers
        elif level == "high":
            buffers = self.high_buffers
        else:
            raise ValueError("level 必須是 'low' 或 'high'。")

        if not 0 <= int(agent_index) < len(buffers):
            raise IndexError("agent_index 超出 ReplayBuffer 範圍。")
        if not self.buffers_ready(level, agent_index):
            raise RuntimeError(
                f"Agent {agent_index} 的 {level}-level buffer 樣本不足。"
            )

        return buffers[int(agent_index)].sample()

    def collect_random_warmup(self):
        """功能：以隨機 goal、move 與 service 填充兩種 buffer，再開始梯度更新。"""
        if (
            len(self.low_buffers) != self.num_uavs
            or len(self.high_buffers) != self.num_uavs
        ):
            self.build_replay_buffers()

        self.rng.switch_mode("train")
        steps_to_collect = max(
            0,
            self.warmup_steps - self.global_step,
        )
        if steps_to_collect == 0:
            self._print_progress(
                "[Warm-up] 已具備足夠步數，略過隨機收集。"
            )
            return {
                "steps": 0,
                "episodes": 0,
                "mean_reward": 0.0,
                "low_buffer_sizes": [
                    len(buffer) for buffer in self.low_buffers
                ],
                "high_buffer_sizes": [
                    len(buffer) for buffer in self.high_buffers
                ],
            }

        self.reset_episode()
        self._print_progress(
            "[Warm-up] 開始收集 "
            f"{steps_to_collect} steps | "
            f"max movement={self.max_movement:g}"
        )
        self.current_goals = np.random.randint(
            0,
            self.num_devices,
            size=self.num_uavs,
            dtype=np.int64,
        )
        self.segment_goals = self.current_goals.copy()

        collected_rewards = []
        episode_count = 1

        for warmup_index in range(steps_to_collect):
            state = self.current_state.copy()
            goals = self.current_goals.copy()

            move_actions = np.stack(
                [
                    sample_random_move(self.max_movement)
                    for _ in range(self.num_uavs)
                ]
            ).astype(np.float32, copy=False)
            moved_state, move_info = self.env.move_step(
                move_actions
            )

            action_masks = self.build_service_action_masks(
                moved_state,
                goals,
            )
            service_actions = np.empty(
                self.num_uavs,
                dtype=np.int64,
            )
            for agent_index in range(self.num_uavs):
                legal_actions = np.flatnonzero(
                    action_masks[agent_index]
                )
                service_actions[agent_index] = np.random.choice(
                    legal_actions
                )

            next_state, done, service_info = self.env.service_step(
                service_actions
            )
            rewards, reward_info = self.calculate_step_rewards(
                state=state,
                moved_state=moved_state,
                next_state=next_state,
                move_info=move_info,
                service_info=service_info,
                move_actions=move_actions,
                goals=goals,
                goal_completed=self.segment_goal_completed,
                return_info=True,
            )
            self.segment_goal_completed |= reward_info[
                "new_goal_completion"
            ]
            next_goals, _ = self.update_high_context(
                rewards=rewards,
                next_state=next_state,
                done=done,
                epsilon=1.0,
            )
            next_action_masks = self.build_service_action_masks(
                next_state,
                next_goals,
            )

            self.add_low_transitions(
                state=state,
                goals=goals,
                move_actions=move_actions,
                moved_state=moved_state,
                service_actions=service_actions,
                rewards=rewards,
                next_state=next_state,
                next_goals=next_goals,
                done=done,
                action_masks=action_masks,
                next_action_masks=next_action_masks,
                executed_moves=move_info["movement"],
            )

            self.current_state = next_state.copy()
            self.terminated = bool(done)
            self.global_step += 1
            collected_rewards.append(float(np.mean(rewards)))

            if done and warmup_index + 1 < steps_to_collect:
                self.reset_episode()
                self.current_goals = np.random.randint(
                    0,
                    self.num_devices,
                    size=self.num_uavs,
                    dtype=np.int64,
                )
                self.segment_goals = self.current_goals.copy()
                episode_count += 1

            completed_steps = warmup_index + 1
            if (
                completed_steps % self.progress_interval == 0
                or completed_steps == steps_to_collect
            ):
                self._print_progress(
                    "[Warm-up] "
                    f"{completed_steps}/{steps_to_collect} | "
                    f"episodes={episode_count} | "
                    f"mean reward={np.mean(collected_rewards):.3f} | "
                    f"low buffers={self._buffer_size_text(self.low_buffers)} | "
                    f"high buffers={self._buffer_size_text(self.high_buffers)}"
                )

        return {
            "steps": steps_to_collect,
            "episodes": episode_count,
            "mean_reward": float(np.mean(collected_rewards)),
            "low_buffer_sizes": [
                len(buffer) for buffer in self.low_buffers
            ],
            "high_buffer_sizes": [
                len(buffer) for buffer in self.high_buffers
            ],
        }

    # ------------------------------------------------------------------
    # Agent 更新
    # ------------------------------------------------------------------
    def should_update_low(self, global_step):
        """功能：依 low-level 更新頻率判斷本 step 是否更新 Actors 與 Critics。"""
        step = int(global_step)
        return (
            step >= self.warmup_steps
            and step % self.low_update_frequency == 0
        )

    def should_update_high(self, global_step):
        """功能：依 warm-up 與 high-level 更新頻率判斷本 step 是否排程 High DQN。"""
        step = int(global_step)
        return (
            step >= self.warmup_steps
            and step % self.high_update_frequency == 0
        )

    def update_agents(self, global_step):
        """功能：為每個 Agent 取樣 high/low batch，呼叫一次 agent.learn() 並彙整 loss。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()
        if (
            len(self.low_buffers) != self.num_uavs
            or len(self.high_buffers) != self.num_uavs
        ):
            self.build_replay_buffers()

        low_scheduled = self.should_update_low(global_step)
        high_scheduled = self.should_update_high(global_step)
        if not low_scheduled and not high_scheduled:
            return {}

        all_losses = {}
        for agent_index, agent in enumerate(self.agents):
            update_low = (
                low_scheduled
                and self.buffers_ready("low", agent_index)
            )
            update_high = (
                high_scheduled
                and self.buffers_ready("high", agent_index)
            )
            if not update_low and not update_high:
                continue

            accumulated_losses = {}
            for _ in range(self.updates_per_step):
                low_experiences = (
                    self.sample_agent_experiences(
                        "low",
                        agent_index,
                    )
                    if update_low
                    else None
                )
                high_experiences = (
                    self.sample_agent_experiences(
                        "high",
                        agent_index,
                    )
                    if update_high
                    else None
                )

                losses = agent.learn(
                    high_experiences=high_experiences,
                    low_experiences=low_experiences,
                    update_high=update_high,
                    update_low=update_low,
                )
                for loss_name, loss_value in losses.items():
                    accumulated_losses.setdefault(
                        loss_name,
                        [],
                    ).append(float(loss_value))

            all_losses[f"agent_{agent_index}"] = {
                loss_name: float(np.mean(values))
                for loss_name, values in accumulated_losses.items()
            }

        return all_losses

    # ------------------------------------------------------------------
    # 訓練、評估與輸出
    # ------------------------------------------------------------------
    def train_episode(self, episode, epsilon):
        """功能：執行一個 episode，處理 goal 邊界、transition、更新及 step 紀錄。"""
        if (
            len(self.low_buffers) != self.num_uavs
            or len(self.high_buffers) != self.num_uavs
        ):
            self.build_replay_buffers()

        self.current_episode = int(episode)
        initial_state = self.reset_episode()
        initial_uav_locations = self.env.uav_locations.copy()
        initial_device_coord = self.env.device_coord.copy()
        initial_risk_regions = self.env.risky_region.copy()

        self.current_goals = self.select_high_goals(
            initial_state,
            epsilon=epsilon,
            deterministic=False,
        )
        self.segment_goals = self.current_goals.copy()

        agent_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        base_agent_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        total_power = 0.0
        total_risk_events = 0
        total_movement_distance = 0.0
        total_wasted_move = 0.0
        total_goal_completions = 0

        while not self.terminated:
            state = self.current_state.copy()
            goals = self.current_goals.copy()
            step_data = self.collect_environment_step(
                state=state,
                goals=goals,
                deterministic=False,
            )

            next_goals, refresh_mask = self.update_high_context(
                rewards=step_data["rewards"],
                next_state=step_data["next_state"],
                done=step_data["done"],
                epsilon=epsilon,
            )
            next_action_masks = self.build_service_action_masks(
                step_data["next_state"],
                next_goals,
            )
            self.add_low_transitions(
                state=state,
                goals=goals,
                move_actions=step_data["move_actions"],
                moved_state=step_data["moved_state"],
                service_actions=step_data["service_actions"],
                rewards=step_data["rewards"],
                next_state=step_data["next_state"],
                next_goals=next_goals,
                done=step_data["done"],
                action_masks=step_data["action_masks"],
                next_action_masks=next_action_masks,
                executed_moves=step_data["move_info"]["movement"],
            )

            self.global_step += 1
            losses = self.update_agents(self.global_step)

            step_data["next_goals"] = next_goals
            step_data["goal_refresh_mask"] = refresh_mask
            self.record_step(
                step_data,
                losses=losses,
            )

            agent_returns += step_data["rewards"]
            base_agent_returns += step_data["reward_info"][
                "base_rewards"
            ]
            total_power += float(
                step_data["service_info"]["total_power"]
            )
            total_risk_events += int(
                np.sum(
                    step_data["move_info"]["risk_triggered"]
                )
            )
            total_movement_distance += float(
                np.sum(step_data["move_info"]["distance"])
            )
            total_wasted_move += float(
                np.sum(step_data["reward_info"]["wasted_move"])
            )
            total_goal_completions += int(
                np.sum(
                    step_data["reward_info"][
                        "new_goal_completion"
                    ]
                )
            )

            if (
                self.env.step_count % self.progress_interval == 0
                or self.terminated
            ):
                self._print_progress(
                    "[Train] "
                    f"episode={self.current_episode}/{self.num_episodes} | "
                    f"step={self.env.step_count}/{self.max_steps} | "
                    f"global={self.global_step} | "
                    f"return={np.mean(base_agent_returns):.3f} | "
                    f"objective={np.mean(agent_returns):.3f} | "
                    f"reward={np.mean(step_data['rewards']):.3f} | "
                    f"{self._loss_progress_text(losses)}"
                )

        episode_data = {
            "episode": self.current_episode,
            "epsilon": float(epsilon),
            "steps": int(self.env.step_count),
            "global_step": int(self.global_step),
            "team_return": float(np.mean(base_agent_returns)),
            "agent_returns": base_agent_returns,
            "shaped_team_return": float(np.mean(agent_returns)),
            "shaped_agent_returns": agent_returns,
            "total_power": total_power,
            "total_risk_events": total_risk_events,
            "total_movement_distance": total_movement_distance,
            "total_wasted_move": total_wasted_move,
            "total_goal_completions": total_goal_completions,
            "final_mean_aoi": float(np.mean(self.env.aoi)),
            "initial_uav_locations": initial_uav_locations,
            "initial_device_coord": initial_device_coord,
            "risk_regions": initial_risk_regions,
        }
        self.record_episode(episode_data)

        return self.episode_logs[-1]

    def train(self):
        """功能：執行 warm-up 與所有 episodes，控制 epsilon、評估週期和最終輸出。"""
        if (
            len(self.low_buffers) != self.num_uavs
            or len(self.high_buffers) != self.num_uavs
        ):
            self.build_replay_buffers()

        self.initialize_logs()
        self.rng.switch_mode("train")
        for agent in self.agents:
            agent.set_train_mode()

        self.training_start_time = time.perf_counter()
        self._print_progress(
            "[HSAC] 開始訓練 | "
            f"device={self.device} | "
            f"episodes={self.num_episodes} | "
            f"steps/episode={self.max_steps} | "
            f"high interval={self.high_interval}"
        )
        warmup_summary = self.collect_random_warmup()
        episode_phase_start = time.perf_counter()

        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.perf_counter()
            epsilon = linear_epsilon(
                step=self.global_step,
                start=self.epsilon_start,
                end=self.epsilon_end,
                decay_steps=self.epsilon_decay_steps,
            )
            episode_result = self.train_episode(
                episode=episode,
                epsilon=epsilon,
            )

            elapsed = time.perf_counter() - episode_phase_start
            episode_elapsed = (
                time.perf_counter() - episode_start_time
            )
            average_episode_time = elapsed / episode
            eta_seconds = average_episode_time * (
                self.num_episodes - episode
            )
            self._print_progress(
                "[Episode] "
                f"{episode}/{self.num_episodes} | "
                f"return={episode_result['team_return']:.3f} | "
                f"epsilon={epsilon:.4f} | "
                f"steps={episode_result['steps']} | "
                f"time={episode_elapsed:.2f}s | "
                f"ETA={self._format_seconds(eta_seconds)}"
            )

            if episode % self.eval_period == 0:
                self.evaluate(self.eval_episodes)

        self.training_end_time = time.perf_counter()
        self._print_progress(
            "[HSAC] 訓練完成 | "
            f"elapsed={self._format_seconds(self.training_end_time - self.training_start_time)} | "
            f"best eval={self.best_eval_return}"
        )

        return {
            "warmup": warmup_summary,
            "episodes": list(self.episode_logs),
            "evaluations": list(self.eval_logs),
            "elapsed_seconds": (
                self.training_end_time
                - self.training_start_time
            ),
        }

    def evaluate_episode(self):
        """功能：以 deterministic high/move/service 策略執行一個不更新參數的 episode。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        # 使用獨立環境，避免 evaluation 改動訓練中的 episode context。
        eval_env = Environment(
            num_devices=self.num_devices,
            num_uavs=self.num_uavs,
            max_steps=self.max_steps,
            max_movement=self.max_movement,
            aoi_max=self.aoi_max,
            energy_weight=self.energy_weight,
            risk_penalty=self.risk_penalty,
            risk_probability=self.risk_probability,
        )
        eval_env.reset()

        state = eval_env.get_state()
        goals = self.select_high_goals(
            state,
            epsilon=0.0,
            deterministic=True,
        )
        goal_durations = np.zeros(
            self.num_uavs,
            dtype=np.int32,
        )
        goal_completed = np.zeros(
            self.num_uavs,
            dtype=np.bool_,
        )
        agent_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        base_agent_returns = np.zeros(
            self.num_uavs,
            dtype=np.float32,
        )
        total_power = 0.0
        total_risk_events = 0
        total_movement_distance = 0.0
        done = False

        while not done:
            move_actions = self.select_move_actions(
                state,
                goals,
                deterministic=True,
            )
            moved_state, move_info = eval_env.move_step(
                move_actions
            )
            action_masks = self.build_service_action_masks(
                moved_state,
                goals,
            )
            service_actions = self.select_service_actions(
                moved_state,
                goals,
                action_masks=action_masks,
                deterministic=True,
            )
            next_state, done, service_info = eval_env.service_step(
                service_actions
            )
            rewards, reward_info = self.calculate_step_rewards(
                state=state,
                moved_state=moved_state,
                next_state=next_state,
                move_info=move_info,
                service_info=service_info,
                move_actions=move_actions,
                goals=goals,
                goal_completed=goal_completed,
                return_info=True,
            )
            goal_completed |= reward_info[
                "new_goal_completion"
            ]

            agent_returns += rewards
            base_agent_returns += reward_info["base_rewards"]
            total_power += float(service_info["total_power"])
            total_risk_events += int(
                np.sum(move_info["risk_triggered"])
            )
            total_movement_distance += float(
                np.sum(move_info["distance"])
            )
            goal_durations += 1

            refresh_mask = self.should_refresh_goal(
                goal_durations,
                done,
            )
            if not done:
                refresh_mask = np.asarray(
                    refresh_mask,
                    dtype=np.bool_,
                )
                candidate_goals = self.select_high_goals(
                    next_state,
                    epsilon=0.0,
                    deterministic=True,
                )
                goals[refresh_mask] = candidate_goals[
                    refresh_mask
                ]
                goal_durations[refresh_mask] = 0
                goal_completed[refresh_mask] = False

            state = next_state

        return {
            "team_return": float(np.mean(base_agent_returns)),
            "shaped_team_return": float(np.mean(agent_returns)),
            "agent_returns": base_agent_returns,
            "shaped_agent_returns": agent_returns,
            "steps": int(eval_env.step_count),
            "total_power": total_power,
            "total_risk_events": total_risk_events,
            "total_movement_distance": total_movement_distance,
            "final_mean_aoi": float(np.mean(eval_env.aoi)),
        }

    def evaluate(self, num_episodes=None):
        """功能：切換 evaluation RNG，執行多個 episode 並回傳平均指標。"""
        episode_count = (
            self.eval_episodes
            if num_episodes is None
            else max(1, int(num_episodes))
        )
        previous_mode = self.rng.mode
        self.rng.switch_mode("eval")
        self._print_progress(
            "[Eval] 開始 | "
            f"training episode={self.current_episode} | "
            f"runs={episode_count}"
        )

        try:
            episode_results = [
                self.evaluate_episode()
                for _ in range(episode_count)
            ]
        finally:
            self.rng.switch_mode(previous_mode)

        team_returns = np.asarray(
            [
                result["team_return"]
                for result in episode_results
            ],
            dtype=np.float32,
        )
        shaped_team_returns = np.asarray(
            [
                result["shaped_team_return"]
                for result in episode_results
            ],
            dtype=np.float32,
        )
        eval_record = {
            "training_episode": int(self.current_episode),
            "global_step": int(self.global_step),
            "num_episodes": episode_count,
            "mean_return": float(np.mean(team_returns)),
            "std_return": float(np.std(team_returns)),
            "episode_returns": team_returns,
            "mean_shaped_return": float(
                np.mean(shaped_team_returns)
            ),
            "shaped_episode_returns": shaped_team_returns,
            "mean_steps": float(
                np.mean(
                    [
                        result["steps"]
                        for result in episode_results
                    ]
                )
            ),
            "mean_total_power": float(
                np.mean(
                    [
                        result["total_power"]
                        for result in episode_results
                    ]
                )
            ),
            "mean_risk_events": float(
                np.mean(
                    [
                        result["total_risk_events"]
                        for result in episode_results
                    ]
                )
            ),
            "mean_final_aoi": float(
                np.mean(
                    [
                        result["final_mean_aoi"]
                        for result in episode_results
                    ]
                )
            ),
        }
        eval_record = self._to_log_value(eval_record)
        self.eval_logs.append(eval_record)

        if (
            self.best_eval_return is None
            or eval_record["mean_return"]
            > self.best_eval_return
        ):
            self.best_eval_return = eval_record["mean_return"]

        self._print_progress(
            "[Eval] 完成 | "
            f"mean return={eval_record['mean_return']:.3f} | "
            f"std={eval_record['std_return']:.3f} | "
            f"mean AOI={eval_record['mean_final_aoi']:.3f}"
        )

        return eval_record

    def record_step(self, step_data, losses=None):
        """功能：保存狀態、goal、兩種 action、reward、AOI、power、risk 與 loss。"""
        step_record = {
            "episode": int(self.current_episode),
            "episode_step": int(self.env.step_count),
            "global_step": int(self.global_step),
            "state": step_data["state"],
            "goals": step_data["goals"],
            "move_actions": step_data["move_actions"],
            "executed_moves": step_data["move_info"]["movement"],
            "moved_state": step_data["moved_state"],
            "service_actions": step_data["service_actions"],
            "rewards": step_data["rewards"],
            "next_state": step_data["next_state"],
            "next_goals": step_data.get("next_goals"),
            "goal_refresh_mask": step_data.get(
                "goal_refresh_mask"
            ),
            "action_masks": step_data["action_masks"],
            "uav_locations": self.env.uav_locations.copy(),
            "device_aoi": self.env.aoi.copy(),
            "power": step_data["service_info"]["power"],
            "total_power": step_data["service_info"][
                "total_power"
            ],
            "mean_aoi": step_data["service_info"]["mean_aoi"],
            "risk_triggered": step_data["move_info"][
                "risk_triggered"
            ],
            "movement_distance": step_data["move_info"][
                "distance"
            ],
            "base_rewards": step_data["reward_info"][
                "base_rewards"
            ],
            "goal_progress": step_data["reward_info"][
                "goal_progress"
            ],
            "new_goal_completion": step_data["reward_info"][
                "new_goal_completion"
            ],
            "wasted_move": step_data["reward_info"][
                "wasted_move"
            ],
            "done": step_data["done"],
        }
        step_record = self._to_log_value(step_record)
        self.step_logs.append(step_record)

        if losses:
            loss_record = {
                "episode": int(self.current_episode),
                "episode_step": int(self.env.step_count),
                "global_step": int(self.global_step),
                "losses": losses,
            }
            self.loss_logs.append(
                self._to_log_value(loss_record)
            )

        return step_record

    def record_episode(self, episode_data):
        """功能：保存 episode return、長度、初始場景及彙總通訊與風險指標。"""
        episode_record = self._to_log_value(
            dict(episode_data)
        )
        self.episode_logs.append(episode_record)
        return episode_record

    @staticmethod
    def _to_log_value(value):
        """將 NumPy、Torch 與巢狀容器轉成可 JSON 序列化的資料。"""
        if isinstance(value, dict):
            return {
                key: HSACOnlineTrainer._to_log_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                HSACOnlineTrainer._to_log_value(item)
                for item in value
            ]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.ndim == 0:
                return tensor.item()
            return tensor.tolist()
        if isinstance(value, (Path, torch.device)):
            return str(value)
        return value

    def _print_progress(self, message):
        """依 quiet 設定輸出可即時觀察的訓練進度。"""
        if self.show_progress:
            print(
                message,
                flush=True,
            )

    @staticmethod
    def _format_seconds(seconds):
        """將秒數格式化為 HH:MM:SS。"""
        seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _buffer_size_text(buffers):
        """將每架 UAV 的 buffer 大小壓縮成單行文字。"""
        return "[" + ", ".join(
            str(len(buffer)) for buffer in buffers
        ) + "]"

    @staticmethod
    def _loss_progress_text(losses):
        """將巢狀 Agent losses 摘要為單一平均絕對值。"""
        loss_values = [
            abs(float(loss_value))
            for agent_losses in losses.values()
            for loss_value in agent_losses.values()
        ]
        if not loss_values:
            return "loss=waiting"
        return f"loss mean(abs)={np.mean(loss_values):.4f}"

    def save_checkpoint(self, checkpoint_name=None):
        """功能：保存所有 Agent、optimizer、計數器與必要的續訓狀態。"""
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        output_directory = Path(self.output_path)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        checkpoint_path = Path(
            checkpoint_name
            if checkpoint_name is not None
            else "hierarchical_checkpoint.pt"
        )
        if checkpoint_path.suffix == "":
            checkpoint_path = checkpoint_path.with_suffix(".pt")
        if not checkpoint_path.is_absolute():
            checkpoint_path = output_directory / checkpoint_path
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        agent_states = []
        for agent in self.agents:
            agent_states.append(
                {
                    "high_network": agent.high_network.state_dict(),
                    "target_high_network": (
                        agent.target_high_network.state_dict()
                    ),
                    "move_actor": agent.move_actor.state_dict(),
                    "service_actor": agent.service_actor.state_dict(),
                    "critic_network_1": (
                        agent.critic_network_1.state_dict()
                    ),
                    "critic_network_2": (
                        agent.critic_network_2.state_dict()
                    ),
                    "target_critic_network_1": (
                        agent.target_critic_network_1.state_dict()
                    ),
                    "target_critic_network_2": (
                        agent.target_critic_network_2.state_dict()
                    ),
                    "high_optimizer": (
                        agent.high_optimizer.state_dict()
                    ),
                    "move_actor_optimizer": (
                        agent.move_actor_optimizer.state_dict()
                    ),
                    "service_actor_optimizer": (
                        agent.service_actor_optimizer.state_dict()
                    ),
                    "critic_1_optimizer": (
                        agent.critic_1_optimizer.state_dict()
                    ),
                    "critic_2_optimizer": (
                        agent.critic_2_optimizer.state_dict()
                    ),
                    "high_update_count": agent.high_update_count,
                    "low_update_count": agent.low_update_count,
                }
            )

        checkpoint = {
            "checkpoint_version": 1,
            "model": "HSACOnlineTrainer",
            "config": self._config_dict(),
            "dimensions": {
                "state_size": self.state_size,
                "num_devices": self.num_devices,
                "num_uavs": self.num_uavs,
            },
            "mvp_contract": self._checkpoint_contract(),
            "trainer": {
                "global_step": self.global_step,
                "current_episode": self.current_episode,
                "best_eval_return": self.best_eval_return,
            },
            "agents": agent_states,
            "logs": {
                "episode_logs": self.episode_logs,
                "step_logs": self.step_logs,
                "eval_logs": self.eval_logs,
                "loss_logs": self.loss_logs,
            },
            "rng": {
                "mode": self.rng.mode,
                "manager_states": self.rng.states,
                "active_state": self._capture_active_rng_state(),
            },
            "replay_buffers_saved": False,
        }
        torch.save(
            checkpoint,
            checkpoint_path,
        )

        self.best_checkpoint_path = str(
            checkpoint_path.resolve()
        )
        return self.best_checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path,
        load_optimizers=True,
        strict_config=True,
    ):
        """載入模型與 Trainer 狀態；ReplayBuffer 依 MVP 契約重新建立。"""
        checkpoint_path = Path(checkpoint_path).resolve()
        if len(self.agents) != self.num_uavs:
            self.build_agents()

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        dimensions = checkpoint["dimensions"]
        expected_dimensions = {
            "state_size": self.state_size,
            "num_devices": self.num_devices,
            "num_uavs": self.num_uavs,
        }
        if dimensions != expected_dimensions:
            raise ValueError(
                "Checkpoint 維度與目前 Trainer 不一致："
                f"{dimensions} != {expected_dimensions}。"
            )
        if len(checkpoint["agents"]) != len(self.agents):
            raise ValueError(
                "Checkpoint 的 Agent 數量與目前 Trainer 不一致。"
            )
        saved_contract = checkpoint.get("mvp_contract")
        current_contract = self._checkpoint_contract()
        if (
            strict_config
            and saved_contract is not None
            and saved_contract != current_contract
        ):
            differences = {
                key: {
                    "checkpoint": saved_contract.get(key),
                    "current": current_contract.get(key),
                }
                for key in current_contract
                if saved_contract.get(key)
                != current_contract.get(key)
            }
            raise ValueError(
                "Checkpoint 的 MVP 環境／網路設定不一致："
                f"{differences}。若確實要跨設定載入，請使用 "
                "strict_config=False。"
            )

        for agent, saved_agent in zip(
            self.agents,
            checkpoint["agents"],
        ):
            agent.high_network.load_state_dict(
                saved_agent["high_network"]
            )
            agent.target_high_network.load_state_dict(
                saved_agent["target_high_network"]
            )
            agent.move_actor.load_state_dict(
                saved_agent["move_actor"]
            )
            agent.service_actor.load_state_dict(
                saved_agent["service_actor"]
            )
            agent.critic_network_1.load_state_dict(
                saved_agent["critic_network_1"]
            )
            agent.critic_network_2.load_state_dict(
                saved_agent["critic_network_2"]
            )
            agent.target_critic_network_1.load_state_dict(
                saved_agent["target_critic_network_1"]
            )
            agent.target_critic_network_2.load_state_dict(
                saved_agent["target_critic_network_2"]
            )

            if load_optimizers:
                optimizer_pairs = (
                    (
                        agent.high_optimizer,
                        saved_agent["high_optimizer"],
                    ),
                    (
                        agent.move_actor_optimizer,
                        saved_agent["move_actor_optimizer"],
                    ),
                    (
                        agent.service_actor_optimizer,
                        saved_agent["service_actor_optimizer"],
                    ),
                    (
                        agent.critic_1_optimizer,
                        saved_agent["critic_1_optimizer"],
                    ),
                    (
                        agent.critic_2_optimizer,
                        saved_agent["critic_2_optimizer"],
                    ),
                )
                for optimizer, optimizer_state in optimizer_pairs:
                    optimizer.load_state_dict(optimizer_state)
                    self._optimizer_to_device(
                        optimizer,
                        self.device,
                    )

            agent.high_update_count = int(
                saved_agent["high_update_count"]
            )
            agent.low_update_count = int(
                saved_agent["low_update_count"]
            )
            agent.set_train_mode()

        trainer_state = checkpoint["trainer"]
        self.global_step = int(trainer_state["global_step"])
        self.current_episode = int(
            trainer_state["current_episode"]
        )
        self.best_eval_return = trainer_state[
            "best_eval_return"
        ]

        saved_logs = checkpoint.get("logs", {})
        self.episode_logs = saved_logs.get(
            "episode_logs",
            [],
        )
        self.step_logs = saved_logs.get(
            "step_logs",
            [],
        )
        self.eval_logs = saved_logs.get(
            "eval_logs",
            [],
        )
        self.loss_logs = saved_logs.get(
            "loss_logs",
            [],
        )

        rng_state = checkpoint.get("rng")
        if rng_state is not None:
            self.rng.states = rng_state["manager_states"]
            self.rng.mode = rng_state["mode"]
            self._restore_active_rng_state(
                rng_state["active_state"]
            )

        self.best_checkpoint_path = str(checkpoint_path)
        return {
            "checkpoint_path": self.best_checkpoint_path,
            "global_step": self.global_step,
            "current_episode": self.current_episode,
            "replay_buffers_loaded": False,
        }

    def _config_dict(self):
        """將 dict 或 argparse Namespace 設定轉成普通 dict。"""
        if isinstance(self.config, dict):
            return dict(self.config)
        return dict(vars(self.config))

    def _checkpoint_contract(self):
        """建立會影響網路結構、環境動力與 reward 的標準化設定。"""
        return {
            "num_devices": self.num_devices,
            "num_uavs": self.num_uavs,
            "state_size": self.state_size,
            "hidden_size": self.hidden_size,
            "max_steps": self.max_steps,
            "max_movement": self.max_movement,
            "aoi_max": self.aoi_max,
            "energy_weight": self.energy_weight,
            "risk_penalty": self.risk_penalty,
            "risk_probability": self.risk_probability,
            "goal_progress_weight": self.goal_progress_weight,
            "wasted_move_weight": self.wasted_move_weight,
            "goal_completion_bonus": self.goal_completion_bonus,
            "critic_state_contract": "moved_state",
            "reward_credit_assignment": "local_power_and_risk",
            "high_interval": self.high_interval,
            "high_gamma": self.high_gamma,
            "low_gamma": self.low_gamma,
        }

    @staticmethod
    def _optimizer_to_device(optimizer, device):
        """將載入的 optimizer tensor 移至目前 Agent device。"""
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)

    @staticmethod
    def _capture_active_rng_state():
        """擷取目前 process 使用中的 NumPy、Python 與 Torch RNG。"""
        return {
            "np": np.random.get_state(),
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _restore_active_rng_state(rng_state):
        """恢復 checkpoint 儲存當下的 process RNG。"""
        np.random.set_state(rng_state["np"])
        random.setstate(rng_state["random"])
        torch.set_rng_state(rng_state["torch"])
        if (
            torch.cuda.is_available()
            and rng_state.get("torch_cuda") is not None
        ):
            torch.cuda.set_rng_state_all(
                rng_state["torch_cuda"]
            )

    def save_training_results(self):
        """功能：輸出 config、CSV、JSON 與最終 checkpoint。"""
        output_directory = Path(self.output_path)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = self.save_checkpoint()
        summary = self.get_training_summary()
        config_path = output_directory / (
            "hierarchical_config.json"
        )
        episode_path = output_directory / (
            "hierarchical_episode_returns.csv"
        )
        eval_path = output_directory / (
            "hierarchical_eval_returns.csv"
        )
        step_path = output_directory / (
            "hierarchical_step_logs.json"
        )
        loss_path = output_directory / (
            "hierarchical_loss_logs.json"
        )

        config_payload = {
            "general": self._config_dict(),
            "trainer": {
                "device": str(self.device),
                "state_size": self.state_size,
                "num_devices": self.num_devices,
                "num_uavs": self.num_uavs,
                "reward_formula": (
                    "-mean_aoi - energy_weight*mean(power) "
                    "- risk_penalty*mean(risk_triggered)"
                ),
                "training_reward_formula": (
                    "-mean_aoi - energy_weight*power_i "
                    "- risk_penalty*risk_i "
                    "+ goal_progress shaping "
                    "- wasted_move shaping "
                    "+ first_goal_completion bonus"
                ),
                "critic_state_contract": "moved_state",
                "replay_buffers_in_checkpoint": False,
            },
            "agent": (
                self.agents[0].get_parameter()
                if self.agents
                else None
            ),
            "summary": summary,
        }
        with config_path.open(
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                self._to_log_value(config_payload),
                config_file,
                ensure_ascii=False,
                indent=2,
            )

        episode_rows = [
            {
                "episode": record["episode"],
                "epsilon": record["epsilon"],
                "steps": record["steps"],
                "global_step": record["global_step"],
                "team_return": record["team_return"],
                "shaped_team_return": record[
                    "shaped_team_return"
                ],
                "agent_returns": json.dumps(
                    record["agent_returns"]
                ),
                "shaped_agent_returns": json.dumps(
                    record["shaped_agent_returns"]
                ),
                "total_power": record["total_power"],
                "total_risk_events": record[
                    "total_risk_events"
                ],
                "total_movement_distance": record[
                    "total_movement_distance"
                ],
                "total_wasted_move": record[
                    "total_wasted_move"
                ],
                "total_goal_completions": record[
                    "total_goal_completions"
                ],
                "final_mean_aoi": record["final_mean_aoi"],
            }
            for record in self.episode_logs
        ]
        self._write_csv(
            episode_path,
            (
                "episode",
                "epsilon",
                "steps",
                "global_step",
                "team_return",
                "shaped_team_return",
                "agent_returns",
                "shaped_agent_returns",
                "total_power",
                "total_risk_events",
                "total_movement_distance",
                "total_wasted_move",
                "total_goal_completions",
                "final_mean_aoi",
            ),
            episode_rows,
        )

        eval_rows = [
            {
                "training_episode": record[
                    "training_episode"
                ],
                "global_step": record["global_step"],
                "num_episodes": record["num_episodes"],
                "mean_return": record["mean_return"],
                "std_return": record["std_return"],
                "episode_returns": json.dumps(
                    record["episode_returns"]
                ),
                "mean_shaped_return": record[
                    "mean_shaped_return"
                ],
                "shaped_episode_returns": json.dumps(
                    record["shaped_episode_returns"]
                ),
                "mean_steps": record["mean_steps"],
                "mean_total_power": record[
                    "mean_total_power"
                ],
                "mean_risk_events": record[
                    "mean_risk_events"
                ],
                "mean_final_aoi": record["mean_final_aoi"],
            }
            for record in self.eval_logs
        ]
        self._write_csv(
            eval_path,
            (
                "training_episode",
                "global_step",
                "num_episodes",
                "mean_return",
                "std_return",
                "episode_returns",
                "mean_shaped_return",
                "shaped_episode_returns",
                "mean_steps",
                "mean_total_power",
                "mean_risk_events",
                "mean_final_aoi",
            ),
            eval_rows,
        )

        with step_path.open(
            "w",
            encoding="utf-8",
        ) as step_file:
            json.dump(
                self.step_logs,
                step_file,
                ensure_ascii=False,
                indent=2,
            )
        with loss_path.open(
            "w",
            encoding="utf-8",
        ) as loss_file:
            json.dump(
                self.loss_logs,
                loss_file,
                ensure_ascii=False,
                indent=2,
            )

        self._print_progress(
            "[Save] 訓練結果已輸出至 "
            f"{output_directory.resolve()}"
        )
        return {
            "config": str(config_path.resolve()),
            "episode_returns": str(episode_path.resolve()),
            "eval_returns": str(eval_path.resolve()),
            "step_logs": str(step_path.resolve()),
            "loss_logs": str(loss_path.resolve()),
            "checkpoint": checkpoint_path,
        }

    def get_training_summary(self):
        """功能：整理訓練時間、更新次數、buffer 大小與最終評估結果。"""
        elapsed_seconds = None
        if (
            self.training_start_time is not None
            and self.training_end_time is not None
        ):
            elapsed_seconds = (
                self.training_end_time
                - self.training_start_time
            )

        return {
            "device": str(self.device),
            "episodes_completed": len(self.episode_logs),
            "global_step": int(self.global_step),
            "elapsed_seconds": elapsed_seconds,
            "final_team_return": (
                self.episode_logs[-1]["team_return"]
                if self.episode_logs
                else None
            ),
            "final_eval_return": (
                self.eval_logs[-1]["mean_return"]
                if self.eval_logs
                else None
            ),
            "best_eval_return": self.best_eval_return,
            "low_buffer_sizes": [
                len(buffer) for buffer in self.low_buffers
            ],
            "high_buffer_sizes": [
                len(buffer) for buffer in self.high_buffers
            ],
            "agent_updates": [
                {
                    "agent": agent_index,
                    "high": agent.high_update_count,
                    "low": agent.low_update_count,
                }
                for agent_index, agent in enumerate(self.agents)
            ],
            "checkpoint": self.best_checkpoint_path,
        }

    @staticmethod
    def _write_csv(path, field_names, rows):
        """使用固定欄位輸出 CSV，即使目前沒有任何資料列。"""
        with Path(path).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(field_names),
            )
            writer.writeheader()
            writer.writerows(rows)


def Train_HSAC_Online(config=None, device=None):
    """功能：建立 HSACOnlineTrainer 並啟動完整 online training。"""
    trainer = HSACOnlineTrainer(
        config=config,
        device=device,
    )
    training_result = trainer.train()
    output_files = trainer.save_training_results()
    summary = trainer.get_training_summary()

    return trainer, {
        "training": training_result,
        "summary": summary,
        "files": output_files,
    }


if __name__ == "__main__":
    cli_config = get_config(args=sys.argv[1:])
    _, cli_result = Train_HSAC_Online(
        config=cli_config,
        device=cli_config.device,
    )
    print(
        json.dumps(
            {
                "summary": cli_result["summary"],
                "files": cli_result["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
