"""階層式 SAC Agent 的介面骨架。

本 Agent 分為三個決策模組：
1. High-level DQN：每隔 n 個 timestep 選擇長期服務目標。
2. MoveActor：根據目前狀態與 high-level goal 產生連續移動動作。
3. SelectActor：根據移動後狀態與 high-level goal 選擇離散服務動作。

低層 MoveActor 與 SelectActor 共用 twin joint Critic；移動後中間態由
Environment 或 Trainer 提供的無副作用 move preview function 建立。
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.nn.utils import clip_grad_norm_

from networks import DDQN, SAC_Continuous_Actor, SAC_Critic, SAC_Discrete_Actor


class HSAC_Agent:
    """協調 high-level DQN 與兩個 low-level SAC actor。"""

    def __init__(
        self,
        state_size,
        num_devices,
        num_uavs,
        move_action_size=2,
        high_interval=5,
        max_movement=2.0,
        hidden_size=256,
        device="cpu",
        high_target_update_interval=100,
        gradient_clip_norm=1.0,
        move_preview_fn=None,
    ):
        """功能：保存維度與超參數，建立三個決策模組、target network 及 optimizer。"""

        # [1. 基本維度] 每個 HSAC_Agent 對應一架 UAV，num_uavs 僅保存環境資訊。
        self.state_size = int(state_size)
        self.num_devices = int(num_devices)
        self.num_uavs = int(num_uavs)
        self.goal_size = self.num_devices
        self.move_action_size = int(move_action_size)
        self.service_action_size = self.num_devices + 1
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)

        # [2. 網路輸入維度] goal 使用 M 維 one-hot，service 使用 M+1 維 one-hot。
        self.move_input_size = self.state_size + self.goal_size
        self.service_input_size = self.state_size + self.goal_size
        self.critic_input_size = (
            self.state_size
            + self.goal_size
            + self.move_action_size
            + self.service_action_size
        )

        # [3. 訓練參數] 高低層分開保存折扣率與 learning rate，方便後續獨立調整。
        self.high_interval = int(high_interval)
        self.max_movement = float(max_movement)
        self.high_gamma = 0.99
        self.low_gamma = 0.99
        self.tau = 1e-3

        self.high_lr = 1e-4
        self.move_actor_lr = 1e-4
        self.service_actor_lr = 1e-4
        self.critic_lr = 1e-4

        # [4. SAC 參數] 連續移動與離散服務分別保留 entropy coefficient。
        self.alpha_move = 0.2
        self.alpha_service = 0.2
        self.log_std_min = -20
        self.log_std_max = 2
        self.high_target_update_interval = max(
            1,
            int(high_target_update_interval),
        )
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.move_preview_fn = move_preview_fn

        # [5. High-level DQN] 從 S 輸出 M 個長期服務目標的 Q-value。
        self.high_network = DDQN(
            state_size=self.state_size,
            action_size=self.goal_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_high_network = DDQN(
            state_size=self.state_size,
            action_size=self.goal_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_high_network.load_state_dict(
            self.high_network.state_dict()
        )

        # [6. Low-level Actors] MoveActor 處理連續移動，SelectActor 處理離散服務。
        self.move_actor = SAC_Continuous_Actor(
            input_size=self.move_input_size,
            action_size=self.move_action_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.service_actor = SAC_Discrete_Actor(
            input_size=self.service_input_size,
            action_size=self.service_action_size,
            layer_size=self.hidden_size,
        ).to(self.device)

        # [7. Joint twin Critic] 以移動後狀態直接評估服務距離及完整低層動作。
        self.critic_network_1 = SAC_Critic(
            input_size=self.critic_input_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.critic_network_2 = SAC_Critic(
            input_size=self.critic_input_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_critic_network_1 = SAC_Critic(
            input_size=self.critic_input_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_critic_network_2 = SAC_Critic(
            input_size=self.critic_input_size,
            layer_size=self.hidden_size,
        ).to(self.device)

        self.target_critic_network_1.load_state_dict(
            self.critic_network_1.state_dict()
        )
        self.target_critic_network_2.load_state_dict(
            self.critic_network_2.state_dict()
        )

        # Target networks 只用於 bootstrap，不保留梯度且固定使用 eval mode。
        for target_network in (
            self.target_high_network,
            self.target_critic_network_1,
            self.target_critic_network_2,
        ):
            target_network.eval()
            for parameter in target_network.parameters():
                parameter.requires_grad_(False)

        # [8. Optimizers] 三個決策模組與兩個 Critic 各自獨立更新。
        self.high_optimizer = optim.Adam(
            self.high_network.parameters(),
            lr=self.high_lr,
        )
        self.move_actor_optimizer = optim.Adam(
            self.move_actor.parameters(),
            lr=self.move_actor_lr,
        )
        self.service_actor_optimizer = optim.Adam(
            self.service_actor.parameters(),
            lr=self.service_actor_lr,
        )
        self.critic_1_optimizer = optim.Adam(
            self.critic_network_1.parameters(),
            lr=self.critic_lr,
        )
        self.critic_2_optimizer = optim.Adam(
            self.critic_network_2.parameters(),
            lr=self.critic_lr,
        )

        # [9. 更新計數] 後續用於控制 high-level target 同步與記錄更新頻率。
        self.high_update_count = 0
        self.low_update_count = 0

    def sample_policy(
        self,
        actor_name,
        actor_input,
        action_mask=None,
        deterministic=False,
    ):
        """功能：統一低層策略取樣入口，依 actor_name 分派到移動或服務策略。"""
        if actor_name == "move":
            actor_input = self._ensure_feature_batch(
                actor_input,
                self.move_input_size,
                "move_actor_input",
            )
            move_mu, move_log_std = self.move_actor(actor_input)
            move_log_std = torch.clamp(
                move_log_std,
                self.log_std_min,
                self.log_std_max,
            )

            move_distribution = Normal(
                move_mu,
                move_log_std.exp(),
            )
            raw_move_action = (
                move_mu
                if deterministic
                else move_distribution.rsample()
            )

            # Radial tanh 將高斯向量映射至移動半徑內，避免逐軸縮放超過 max_movement。
            raw_move_norm = torch.linalg.vector_norm(
                raw_move_action,
                dim=-1,
                keepdim=True,
            )
            squashed_move_norm = torch.tanh(raw_move_norm)
            move_direction = raw_move_action / (
                raw_move_norm + 1e-6
            )
            move_action = (
                move_direction
                * squashed_move_norm
                * self.max_movement
            )

            # 扣除 radial tanh 及移動半徑縮放的 Jacobian。
            base_log_prob = move_distribution.log_prob(
                raw_move_action
            ).sum(dim=-1, keepdim=True)
            move_scale = torch.as_tensor(
                self.max_movement,
                dtype=move_action.dtype,
                device=self.device,
            )
            log_radial_ratio = (
                torch.log(squashed_move_norm + 1e-6)
                - torch.log(raw_move_norm + 1e-6)
            )
            log_jacobian = (
                self.move_action_size
                * torch.log(move_scale + 1e-6)
                + torch.log(
                    1.0 - squashed_move_norm.pow(2) + 1e-6
                )
                + (self.move_action_size - 1) * log_radial_ratio
            )
            move_log_prob = base_log_prob - log_jacobian

            return move_action, move_log_prob

        if actor_name == "service":
            actor_input = self._ensure_feature_batch(
                actor_input,
                self.service_input_size,
                "service_actor_input",
            )
            service_logits = self.service_actor(actor_input)
            service_logits = self.apply_action_mask(
                service_logits,
                action_mask,
            )

            service_log_probs = F.log_softmax(
                service_logits,
                dim=-1,
            )
            service_probs = service_log_probs.exp()

            if deterministic:
                service_action = torch.argmax(
                    service_logits,
                    dim=-1,
                )
            else:
                service_distribution = Categorical(
                    probs=service_probs
                )
                service_action = service_distribution.sample()

            return (
                service_action,
                service_probs,
                service_log_probs,
            )

        raise ValueError(
            "actor_name 必須為 'move' 或 'service'。"
        )

    def get_action(
        self,
        state,
        module,
        goal=None,
        epsilon=0.0,
        action_mask=None,
        deterministic=False,
    ):
        """功能：提供 Trainer 統一動作入口，依 module 取得 high、move 或 service 動作。"""
        module = str(module).lower()

        if module == "high":
            return self.get_high_action(
                state,
                epsilon=epsilon,
                deterministic=deterministic,
            )

        if goal is None:
            raise ValueError(
                "move 與 service 模組必須提供 high-level goal。"
            )

        if module == "move":
            return self.get_move_action(
                state,
                goal,
                deterministic=deterministic,
            )

        if module == "service":
            return self.get_service_action(
                state,
                goal,
                action_mask=action_mask,
                deterministic=deterministic,
            )

        raise ValueError(
            "module 必須為 'high'、'move' 或 'service'。"
        )

    def learn(
        self,
        high_experiences=None,
        low_experiences=None,
        update_high=True,
        update_low=True,
    ):
        """功能：協調高層與低層更新，並彙整各網路的 loss 與訓練資訊。"""
        losses = {}
        will_update_high = (
            update_high and high_experiences is not None
        )
        will_update_low = (
            update_low and low_experiences is not None
        )

        if not will_update_high and not will_update_low:
            return losses

        # 先確認低層 dynamics 契約，避免高層更新後才因缺少 preview 半途失敗。
        if will_update_low and self.move_preview_fn is None:
            raise RuntimeError(
                "低層 learn 前必須提供 move_preview_fn。"
            )

        self.set_train_mode()

        if will_update_high:
            losses.update(self.learn_high(high_experiences))

        if will_update_low:
            losses.update(self.learn_low(low_experiences))

        return losses

    def select_high_goal(
        self,
        states,
        epsilon=0.0,
        deterministic=False,
    ):
        """功能：以 high-level DQN 的 epsilon-greedy 規則選擇長期裝置目標。"""
        state_batch = self._ensure_feature_batch(
            states,
            self.state_size,
            "states",
        )
        epsilon = min(max(float(epsilon), 0.0), 1.0)

        with torch.no_grad():
            high_q_values = self.high_network(state_batch)
            greedy_goals = torch.argmax(
                high_q_values,
                dim=-1,
            )

            if deterministic or epsilon == 0.0:
                return greedy_goals

            random_goals = torch.randint(
                low=0,
                high=self.goal_size,
                size=(state_batch.shape[0],),
                device=self.device,
            )
            explore = torch.rand(
                state_batch.shape[0],
                device=self.device,
            ) < epsilon

        return torch.where(
            explore,
            random_goals,
            greedy_goals,
        )

    def sample_move_policy(
        self,
        states,
        goals,
        deterministic=False,
    ):
        """功能：由 MoveActor 取樣連續動作，並計算 tanh 修正後的 log probability。"""
        move_input = self.build_move_input(
            states,
            goals,
        )

        return self.sample_policy(
            actor_name="move",
            actor_input=move_input,
            deterministic=deterministic,
        )

    def sample_service_policy(
        self,
        moved_states,
        goals,
        action_mask=None,
        deterministic=False,
    ):
        """功能：由 SelectActor 對 M+1 個服務選項取樣，並套用合法動作 mask。"""
        service_input = self.build_service_input(
            moved_states,
            goals,
        )

        return self.sample_policy(
            actor_name="service",
            actor_input=service_input,
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def get_high_action(
        self,
        state,
        epsilon=0.0,
        deterministic=False,
    ):
        """功能：將單筆 state 轉為 batch 後取得 high-level goal，供 n 步內持續使用。"""
        was_training = self.high_network.training
        self.high_network.eval()

        goal = self.select_high_goal(
            state,
            epsilon=epsilon,
            deterministic=deterministic,
        )

        if was_training:
            self.high_network.train()

        goal = goal.cpu().numpy()
        if goal.shape[0] == 1:
            return int(goal[0])

        return goal

    def get_move_action(
        self,
        state,
        goal,
        deterministic=False,
    ):
        """功能：取得可直接交給 Environment.move_step() 的連續移動動作。"""
        was_training = self.move_actor.training
        self.move_actor.eval()

        with torch.no_grad():
            move_action, _ = self.sample_move_policy(
                state,
                goal,
                deterministic=deterministic,
            )

        if was_training:
            self.move_actor.train()

        move_action = move_action.cpu().numpy()
        if move_action.shape[0] == 1:
            return move_action[0]

        return move_action

    def get_service_action(
        self,
        moved_state,
        goal,
        action_mask=None,
        deterministic=False,
    ):
        """功能：取得可直接交給 Environment.service_step() 的離散服務動作。"""
        was_training = self.service_actor.training
        self.service_actor.eval()

        with torch.no_grad():
            service_action, _, _ = self.sample_service_policy(
                moved_state,
                goal,
                action_mask=action_mask,
                deterministic=deterministic,
            )

        if was_training:
            self.service_actor.train()

        service_action = service_action.cpu().numpy()
        if service_action.shape[0] == 1:
            return int(service_action[0])

        return service_action

    def goal_to_onehot(self, goals):
        """功能：將 high-level 裝置索引轉成 M 維 one-hot goal。"""
        goal_tensor = self.to_tensor(goals)

        # 已是浮點 one-hot 時直接沿用，避免重複編碼。
        if (
            self._is_encoded_vector(
                goal_tensor,
                self.goal_size,
            )
        ):
            return goal_tensor

        goal_indices = self.to_tensor(goals, dtype=torch.long)
        if goal_indices.dim() > 0 and goal_indices.shape[-1] == 1:
            goal_indices = goal_indices.squeeze(-1)

        return F.one_hot(
            goal_indices,
            num_classes=self.goal_size,
        ).float()

    def service_to_onehot(self, service_actions):
        """功能：將服務索引轉成 M+1 維 one-hot，供 joint Critic 使用。"""
        service_tensor = self.to_tensor(service_actions)

        # SelectActor 的 M+1 維浮點輸出若已編碼完成，便直接沿用。
        if (
            self._is_encoded_vector(
                service_tensor,
                self.service_action_size,
            )
        ):
            return service_tensor

        service_indices = self.to_tensor(
            service_actions,
            dtype=torch.long,
        )
        if (
            service_indices.dim() > 0
            and service_indices.shape[-1] == 1
        ):
            service_indices = service_indices.squeeze(-1)

        return F.one_hot(
            service_indices,
            num_classes=self.service_action_size,
        ).float()

    def build_move_input(self, states, goals):
        """功能：拼接 S 與 goal，建立 MoveActor 的輸入。"""
        state_batch = self._ensure_feature_batch(
            states,
            self.state_size,
            "states",
        )
        goal_batch = self._ensure_feature_batch(
            self.goal_to_onehot(goals),
            self.goal_size,
            "goals",
        )
        state_batch, goal_batch = self._align_batch_sizes(
            state_batch,
            goal_batch,
        )

        return torch.cat((state_batch, goal_batch), dim=-1)

    def build_service_input(self, moved_states, goals):
        """功能：拼接移動後狀態 S' 與 goal，建立 SelectActor 的輸入。"""
        moved_state_batch = self._ensure_feature_batch(
            moved_states,
            self.state_size,
            "moved_states",
        )
        goal_batch = self._ensure_feature_batch(
            self.goal_to_onehot(goals),
            self.goal_size,
            "goals",
        )
        moved_state_batch, goal_batch = self._align_batch_sizes(
            moved_state_batch,
            goal_batch,
        )

        return torch.cat((moved_state_batch, goal_batch), dim=-1)

    def build_joint_critic_input(
        self,
        moved_states,
        goals,
        move_actions,
        service_actions,
    ):
        """功能：拼接移動後狀態、goal、move 與 service，讓 Critic 直接看到服務距離。"""
        moved_state_batch = self._ensure_feature_batch(
            moved_states,
            self.state_size,
            "moved_states",
        )
        goal_batch = self._ensure_feature_batch(
            self.goal_to_onehot(goals),
            self.goal_size,
            "goals",
        )
        move_batch = self._ensure_feature_batch(
            move_actions,
            self.move_action_size,
            "move_actions",
        )
        service_batch = self._ensure_feature_batch(
            self.service_to_onehot(service_actions),
            self.service_action_size,
            "service_actions",
        )

        (
            moved_state_batch,
            goal_batch,
            move_batch,
            service_batch,
        ) = self._align_batch_sizes(
            moved_state_batch,
            goal_batch,
            move_batch,
            service_batch,
        )

        return torch.cat(
            (
                moved_state_batch,
                goal_batch,
                move_batch,
                service_batch,
            ),
            dim=-1,
        )

    def critic_all_services(
        self,
        critic,
        moved_states,
        goals,
        move_actions,
        action_mask=None,
    ):
        """功能：枚舉 M+1 個服務動作的 Q-value，供離散 SAC 期望值更新使用。"""
        moved_state_batch = self._ensure_feature_batch(
            moved_states,
            self.state_size,
            "moved_states",
        )
        goal_batch = self._ensure_feature_batch(
            self.goal_to_onehot(goals),
            self.goal_size,
            "goals",
        )
        move_batch = self._ensure_feature_batch(
            move_actions,
            self.move_action_size,
            "move_actions",
        )
        moved_state_batch, goal_batch, move_batch = self._align_batch_sizes(
            moved_state_batch,
            goal_batch,
            move_batch,
        )

        batch_size = moved_state_batch.shape[0]
        num_services = self.service_action_size

        # 每筆 S'、goal 與 move 複製 M+1 次，分別搭配一個服務動作。
        expanded_states = moved_state_batch.unsqueeze(1).expand(
            -1,
            num_services,
            -1,
        )
        expanded_goals = goal_batch.unsqueeze(1).expand(
            -1,
            num_services,
            -1,
        )
        expanded_moves = move_batch.unsqueeze(1).expand(
            -1,
            num_services,
            -1,
        )

        all_services = torch.eye(
            num_services,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        # 攤平成 B*(M+1) 筆 Critic 輸入，再還原成每筆資料的所有 Q-value。
        critic_input = self.build_joint_critic_input(
            expanded_states.reshape(-1, self.state_size),
            expanded_goals.reshape(-1, self.goal_size),
            expanded_moves.reshape(-1, self.move_action_size),
            all_services.reshape(-1, num_services),
        )
        q_values = critic(critic_input).reshape(
            batch_size,
            num_services,
        )

        # 非法服務動作使用與 logits 相同的 mask 規則排除。
        if action_mask is not None:
            q_values = self.apply_action_mask(
                q_values,
                action_mask,
            )

        return q_values

    def learn_high(self, experiences):
        """功能：使用高層 transition 更新 online DQN，並計算 duration-aware target。"""
        data = self._prepare_high_experiences(experiences)
        high_targets = self.compute_high_target(data)

        predicted_q = self.high_network(data["states"]).gather(
            1,
            data["goals"].unsqueeze(-1),
        )
        high_loss = F.smooth_l1_loss(
            predicted_q,
            high_targets,
        )

        self.high_optimizer.zero_grad()
        high_loss.backward()
        clip_grad_norm_(
            self.high_network.parameters(),
            self.gradient_clip_norm,
        )
        self.high_optimizer.step()

        self.high_update_count += 1
        if (
            self.high_update_count
            % self.high_target_update_interval
            == 0
        ):
            self.hard_update(
                self.high_network,
                self.target_high_network,
            )

        return {
            "high_loss": high_loss.detach().item(),
        }

    def learn_low(self, experiences):
        """功能：依序協調 joint Critic、MoveActor 與 SelectActor 的低層更新。"""
        data = self._prepare_low_experiences(experiences)

        losses = self.update_low_critics(data)
        losses.update(
            self.update_move_actor(
                data["states"],
                data["goals"],
                action_mask=data["action_masks"],
            )
        )
        losses.update(
            self.update_service_actor(
                data["states"],
                data["moved_states"],
                data["goals"],
                data["move_actions"],
                action_mask=data["action_masks"],
            )
        )

        self.low_update_count += 1
        self.update_target_networks(update_high=False)

        return losses

    def update_low_critics(self, experiences):
        """功能：以完整低層 transition 更新兩個 joint Critic。"""
        data = self._prepare_low_experiences(experiences)
        low_targets = self.compute_low_target(data)
        critic_input = self.build_joint_critic_input(
            data["moved_states"],
            data["goals"],
            data["move_actions"],
            data["service_actions"],
        )

        current_q_1 = self.critic_network_1(critic_input)
        current_q_2 = self.critic_network_2(critic_input)
        critic_1_loss = F.mse_loss(
            current_q_1,
            low_targets,
        )
        critic_2_loss = F.mse_loss(
            current_q_2,
            low_targets,
        )

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        (critic_1_loss + critic_2_loss).backward()
        clip_grad_norm_(
            self.critic_network_1.parameters(),
            self.gradient_clip_norm,
        )
        clip_grad_norm_(
            self.critic_network_2.parameters(),
            self.gradient_clip_norm,
        )
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        return {
            "critic_1_loss": critic_1_loss.detach().item(),
            "critic_2_loss": critic_2_loss.detach().item(),
        }

    def update_move_actor(
        self,
        states,
        goals,
        action_mask=None,
    ):
        """功能：透過 joint Critic 更新 MoveActor 的連續移動策略。"""
        move_actions, move_log_probs = self.sample_move_policy(
            states,
            goals,
            deterministic=False,
        )
        moved_states = self._preview_move_states(
            states,
            move_actions,
        )

        # 凍結 SelectActor 權重，但保留 moved_state -> service probability
        # 對 move action 的梯度，使移動策略能感知位置如何改變後續服務。
        self._set_service_actor_grad(False)
        try:
            (
                _,
                service_probs,
                service_log_probs,
            ) = self.sample_service_policy(
                moved_states,
                goals,
                action_mask=action_mask,
                deterministic=False,
            )
        finally:
            self._set_service_actor_grad(True)

        self._set_online_critic_grad(False)
        try:
            q_1_all = self.critic_all_services(
                self.critic_network_1,
                moved_states,
                goals,
                move_actions,
            )
            q_2_all = self.critic_all_services(
                self.critic_network_2,
                moved_states,
                goals,
                move_actions,
            )
            min_q_all = torch.minimum(q_1_all, q_2_all)

            expected_service_objective = (
                service_probs
                * (
                    self.alpha_service * service_log_probs
                    - min_q_all
                )
            ).sum(dim=-1, keepdim=True)
            move_actor_loss = (
                self.alpha_move * move_log_probs
                + expected_service_objective
            ).mean()

            self.move_actor_optimizer.zero_grad()
            move_actor_loss.backward()
            clip_grad_norm_(
                self.move_actor.parameters(),
                self.gradient_clip_norm,
            )
            self.move_actor_optimizer.step()
        finally:
            self._set_online_critic_grad(True)

        return {
            "move_actor_loss": move_actor_loss.detach().item(),
        }

    def update_service_actor(
        self,
        states,
        moved_states,
        goals,
        move_actions,
        action_mask=None,
    ):
        """功能：透過所有服務動作的期望 Q-value 更新 SelectActor。"""
        (
            _,
            service_probs,
            service_log_probs,
        ) = self.sample_service_policy(
            moved_states,
            goals,
            action_mask=action_mask,
            deterministic=False,
        )

        # 枚舉 Q-value 不依賴 SelectActor，因此不建立 Critic 的梯度圖。
        with torch.no_grad():
            q_1_all = self.critic_all_services(
                self.critic_network_1,
                moved_states,
                goals,
                move_actions,
            )
            q_2_all = self.critic_all_services(
                self.critic_network_2,
                moved_states,
                goals,
                move_actions,
            )
            min_q_all = torch.minimum(q_1_all, q_2_all)

        service_actor_loss = (
            service_probs
            * (
                self.alpha_service * service_log_probs
                - min_q_all
            )
        ).sum(dim=-1).mean()

        self.service_actor_optimizer.zero_grad()
        service_actor_loss.backward()
        clip_grad_norm_(
            self.service_actor.parameters(),
            self.gradient_clip_norm,
        )
        self.service_actor_optimizer.step()

        return {
            "service_actor_loss": service_actor_loss.detach().item(),
        }

    def compute_high_target(self, experiences):
        """功能：計算高層 DQN 的 n-step 或實際 duration 折扣 target。"""
        data = self._prepare_high_experiences(experiences)

        with torch.no_grad():
            # Double DQN：online network 選 action，target network 評估。
            next_goals = torch.argmax(
                self.high_network(data["next_states"]),
                dim=-1,
                keepdim=True,
            )
            next_q = self.target_high_network(
                data["next_states"]
            ).gather(1, next_goals)

            duration_discount = torch.pow(
                torch.full_like(
                    data["durations"],
                    self.high_gamma,
                ),
                data["durations"],
            )
            # cumulative_rewards 應由 Trainer 保存為區段內已折扣回報。
            high_targets = (
                data["cumulative_rewards"]
                + (1.0 - data["dones"])
                * duration_discount
                * next_q
            )

        return high_targets

    def compute_low_target(self, experiences):
        """功能：使用 target twin Critic 與下一步策略計算低層 soft Q target。"""
        data = self._prepare_low_experiences(experiences)

        with torch.no_grad():
            (
                next_move_actions,
                next_move_log_probs,
            ) = self.sample_move_policy(
                data["next_states"],
                data["next_goals"],
                deterministic=False,
            )
            next_moved_states = self._preview_move_states(
                data["next_states"],
                next_move_actions,
            )
            (
                _,
                next_service_probs,
                next_service_log_probs,
            ) = self.sample_service_policy(
                next_moved_states,
                data["next_goals"],
                action_mask=data["next_action_masks"],
                deterministic=False,
            )

            target_q_1_all = self.critic_all_services(
                self.target_critic_network_1,
                next_moved_states,
                data["next_goals"],
                next_move_actions,
            )
            target_q_2_all = self.critic_all_services(
                self.target_critic_network_2,
                next_moved_states,
                data["next_goals"],
                next_move_actions,
            )
            target_min_q_all = torch.minimum(
                target_q_1_all,
                target_q_2_all,
            )

            next_service_value = (
                next_service_probs
                * (
                    target_min_q_all
                    - self.alpha_service
                    * next_service_log_probs
                )
            ).sum(dim=-1, keepdim=True)
            next_soft_value = (
                next_service_value
                - self.alpha_move * next_move_log_probs
            )
            low_targets = (
                data["rewards"]
                + self.low_gamma
                * (1.0 - data["dones"])
                * next_soft_value
            )

        return low_targets

    def _prepare_high_experiences(self, experiences):
        """功能：將高層 tuple、namedtuple 或 dict 統一成可訓練的 tensor batch。"""
        states = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("states", "state"),
                0,
            ),
            self.state_size,
            "high_states",
        )
        goals = self._ensure_index_batch(
            self._experience_field(
                experiences,
                ("goals", "goal"),
                1,
            ),
            self.goal_size,
            "high_goals",
        )
        cumulative_rewards = self._ensure_column_batch(
            self._experience_field(
                experiences,
                ("cumulative_rewards", "cumulative_reward"),
                2,
            ),
            "cumulative_rewards",
        )
        next_states = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("next_states", "next_state"),
                3,
            ),
            self.state_size,
            "high_next_states",
        )
        dones = self._ensure_column_batch(
            self._experience_field(
                experiences,
                ("dones", "done"),
                4,
            ),
            "high_dones",
        )
        durations = self._ensure_column_batch(
            self._experience_field(
                experiences,
                ("durations", "duration"),
                5,
            ),
            "high_durations",
        )

        batch_size = states.shape[0]
        return {
            "states": states,
            "goals": self._expand_experience_batch(
                goals,
                batch_size,
                "high_goals",
            ),
            "cumulative_rewards": self._expand_experience_batch(
                cumulative_rewards,
                batch_size,
                "cumulative_rewards",
            ),
            "next_states": self._expand_experience_batch(
                next_states,
                batch_size,
                "high_next_states",
            ),
            "dones": self._expand_experience_batch(
                dones,
                batch_size,
                "high_dones",
            ),
            "durations": self._expand_experience_batch(
                durations,
                batch_size,
                "high_durations",
            ),
        }

    def _prepare_low_experiences(self, experiences):
        """功能：統一低層 transition 的 shape、dtype、device 與可選 action mask。"""
        states = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("states", "state"),
                0,
            ),
            self.state_size,
            "low_states",
        )
        goals = self._ensure_index_batch(
            self._experience_field(
                experiences,
                ("goals", "goal"),
                1,
            ),
            self.goal_size,
            "low_goals",
        )
        move_actions = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("move_actions", "move_action"),
                2,
            ),
            self.move_action_size,
            "move_actions",
        )
        executed_move_values = self._experience_field(
            experiences,
            ("executed_moves", "executed_move"),
            None,
            required=False,
        )
        executed_moves = (
            move_actions
            if executed_move_values is None
            else self._ensure_feature_batch(
                executed_move_values,
                self.move_action_size,
                "executed_moves",
            )
        )
        moved_states = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("moved_states", "moved_state"),
                3,
            ),
            self.state_size,
            "moved_states",
        )
        service_actions = self._ensure_index_batch(
            self._experience_field(
                experiences,
                ("service_actions", "service_action"),
                4,
            ),
            self.service_action_size,
            "service_actions",
        )
        rewards = self._ensure_column_batch(
            self._experience_field(
                experiences,
                ("rewards", "reward"),
                5,
            ),
            "low_rewards",
        )
        next_states = self._ensure_feature_batch(
            self._experience_field(
                experiences,
                ("next_states", "next_state"),
                6,
            ),
            self.state_size,
            "low_next_states",
        )
        next_goals = self._ensure_index_batch(
            self._experience_field(
                experiences,
                ("next_goals", "next_goal"),
                7,
            ),
            self.goal_size,
            "next_goals",
        )
        dones = self._ensure_column_batch(
            self._experience_field(
                experiences,
                ("dones", "done"),
                8,
            ),
            "low_dones",
        )

        action_masks = self._experience_field(
            experiences,
            ("action_masks", "action_mask"),
            None,
            required=False,
        )
        next_action_masks = self._experience_field(
            experiences,
            ("next_action_masks", "next_action_mask"),
            None,
            required=False,
        )

        batch_size = states.shape[0]
        prepared = {
            "states": states,
            "goals": self._expand_experience_batch(
                goals,
                batch_size,
                "low_goals",
            ),
            "move_actions": self._expand_experience_batch(
                move_actions,
                batch_size,
                "move_actions",
            ),
            "executed_moves": self._expand_experience_batch(
                executed_moves,
                batch_size,
                "executed_moves",
            ),
            "moved_states": self._expand_experience_batch(
                moved_states,
                batch_size,
                "moved_states",
            ),
            "service_actions": self._expand_experience_batch(
                service_actions,
                batch_size,
                "service_actions",
            ),
            "rewards": self._expand_experience_batch(
                rewards,
                batch_size,
                "low_rewards",
            ),
            "next_states": self._expand_experience_batch(
                next_states,
                batch_size,
                "low_next_states",
            ),
            "next_goals": self._expand_experience_batch(
                next_goals,
                batch_size,
                "next_goals",
            ),
            "dones": self._expand_experience_batch(
                dones,
                batch_size,
                "low_dones",
            ),
            "action_masks": self._prepare_action_mask_batch(
                action_masks,
                batch_size,
                "action_masks",
            ),
            "next_action_masks": self._prepare_action_mask_batch(
                next_action_masks,
                batch_size,
                "next_action_masks",
            ),
        }

        return prepared

    def _experience_field(
        self,
        experiences,
        names,
        index,
        required=True,
    ):
        """功能：從 dict、namedtuple 或 tuple 讀取同一個 experience 欄位。"""
        if isinstance(experiences, dict):
            for name in names:
                if name in experiences:
                    return experiences[name]
        else:
            for name in names:
                if hasattr(experiences, name):
                    return getattr(experiences, name)

            if index is not None:
                try:
                    return experiences[index]
                except (IndexError, TypeError):
                    pass

        if required:
            raise KeyError(
                f"experiences 缺少必要欄位：{' 或 '.join(names)}"
            )

        return None

    def _ensure_index_batch(self, value, action_size, name):
        """功能：將 index 或 one-hot action 統一成一維 long batch。"""
        value_tensor = self.to_tensor(value)

        if self._is_encoded_vector(value_tensor, action_size):
            index_tensor = torch.argmax(
                value_tensor,
                dim=-1,
            )
        else:
            index_tensor = self.to_tensor(
                value,
                dtype=torch.long,
            )
            if (
                index_tensor.dim() > 0
                and index_tensor.shape[-1] == 1
            ):
                index_tensor = index_tensor.squeeze(-1)

        if index_tensor.dim() == 0:
            index_tensor = index_tensor.unsqueeze(0)

        if index_tensor.dim() != 1:
            raise ValueError(
                f"{name} 必須是 scalar、(batch,) 或合法 one-hot，"
                f"目前為 {tuple(index_tensor.shape)}。"
            )

        return index_tensor.long()

    def _ensure_column_batch(self, value, name):
        """功能：將 reward、done 或 duration 統一成 (batch, 1) float tensor。"""
        value_tensor = self.to_tensor(value)

        if value_tensor.dim() == 0:
            value_tensor = value_tensor.reshape(1, 1)
        elif value_tensor.dim() == 1:
            value_tensor = value_tensor.unsqueeze(-1)

        if value_tensor.dim() != 2 or value_tensor.shape[-1] != 1:
            raise ValueError(
                f"{name} 的 shape 應為 (batch,) 或 (batch, 1)，"
                f"目前為 {tuple(value_tensor.shape)}。"
            )

        return value_tensor

    def _expand_experience_batch(self, value, batch_size, name):
        """功能：讓單筆 experience 欄位可展開，並拒絕不一致的 batch size。"""
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(
                batch_size,
                *value.shape[1:],
            )

        raise ValueError(
            f"{name} 的 batch size={value.shape[0]}，"
            f"但 states 的 batch size={batch_size}。"
        )

    def _prepare_action_mask_batch(
        self,
        action_mask,
        batch_size,
        name,
    ):
        """功能：整理可選的 service mask；True 表示合法，None 表示全合法。"""
        if action_mask is None:
            return None

        action_mask = self.to_tensor(
            action_mask,
            dtype=torch.bool,
        )
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        if (
            action_mask.dim() != 2
            or action_mask.shape[-1]
            != self.service_action_size
        ):
            raise ValueError(
                f"{name} 的 shape 應為 ({self.service_action_size},) "
                f"或 (batch, {self.service_action_size})。"
            )

        return self._expand_experience_batch(
            action_mask,
            batch_size,
            name,
        )

    def _preview_move_states(self, states, move_actions):
        """功能：以外部無副作用 dynamics 將 state 與 move 轉成 SelectActor 中間態。"""
        if self.move_preview_fn is None:
            raise RuntimeError(
                "低層學習需要 move_preview_fn(states, move_actions)，"
                "不可直接以 next_state 代替移動後中間態。"
            )

        moved_states = self.move_preview_fn(
            states.detach(),
            move_actions,
        )
        moved_states = self._ensure_feature_batch(
            moved_states,
            self.state_size,
            "preview_moved_states",
        )

        return self._expand_experience_batch(
            moved_states,
            states.shape[0],
            "preview_moved_states",
        )

    def _set_online_critic_grad(self, enabled):
        """功能：更新 Actor 時凍結 Critic 權重，但保留 Q 對 move action 的梯度。"""
        for critic in (
            self.critic_network_1,
            self.critic_network_2,
        ):
            for parameter in critic.parameters():
                parameter.requires_grad_(enabled)

    def _set_service_actor_grad(self, enabled):
        """功能：更新 MoveActor 時固定服務網路權重，但保留輸入狀態的梯度。"""
        for parameter in self.service_actor.parameters():
            parameter.requires_grad_(enabled)

    def apply_action_mask(self, logits, action_mask):
        """功能：將非法服務動作從 SelectActor 的 categorical 分布中排除。"""
        logits = self.to_tensor(logits)
        if action_mask is None:
            return logits

        valid_actions = self.to_tensor(
            action_mask,
            dtype=torch.bool,
        )

        # 單筆 mask 可套用到整個 batch；True 表示該服務動作合法。
        if logits.dim() == 2 and valid_actions.dim() == 1:
            valid_actions = valid_actions.unsqueeze(0).expand(
                logits.shape[0],
                -1,
            )
        elif (
            logits.dim() == 2
            and valid_actions.dim() == 2
            and valid_actions.shape[0] == 1
        ):
            valid_actions = valid_actions.expand(
                logits.shape[0],
                -1,
            )
        elif logits.dim() == 1 and valid_actions.dim() == 2:
            if valid_actions.shape[0] == 1:
                valid_actions = valid_actions.squeeze(0)

        if logits.shape != valid_actions.shape:
            raise ValueError(
                "action_mask 必須與 logits 同形狀，或是一個可套用到 "
                "整個 batch 的一維 mask。"
            )

        valid_actions = valid_actions.clone()

        # 若整列都不合法，保留最後一個「不服務」動作，避免 softmax 產生 NaN。
        if valid_actions.dim() == 1:
            if not torch.any(valid_actions):
                valid_actions[-1] = True
        else:
            no_valid_action = ~torch.any(
                valid_actions,
                dim=-1,
            )
            valid_actions[no_valid_action, -1] = True

        return logits.masked_fill(
            ~valid_actions,
            torch.finfo(logits.dtype).min,
        )

    def to_tensor(self, value, dtype=torch.float32):
        """功能：統一將 numpy 或 tensor 資料轉到 Agent 使用的 device 與 dtype。"""
        # Tensor 使用 .to() 可保留既有 computation graph。
        if torch.is_tensor(value):
            return value.to(
                device=self.device,
                dtype=dtype,
            )

        return torch.as_tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    def _is_encoded_vector(self, value, vector_size):
        """功能：辨識最後一維為合法 one-hot 或機率向量，避免誤判浮點 index batch。"""
        if (
            value.dim() < 1
            or value.shape[-1] != vector_size
            or not value.is_floating_point()
        ):
            return False

        detached_value = value.detach()
        vector_sums = detached_value.sum(dim=-1)

        return bool(
            torch.all(torch.isfinite(detached_value))
            and torch.all(detached_value >= 0.0)
            and torch.allclose(
                vector_sums,
                torch.ones_like(vector_sums),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def _ensure_feature_batch(self, value, feature_size, name):
        """功能：將單筆特徵補成 batch，並檢查最後一維是否符合網路契約。"""
        value = self.to_tensor(value)

        if value.dim() == 0:
            value = value.reshape(1, 1)
        elif value.dim() == 1:
            value = value.unsqueeze(0)

        if value.dim() != 2 or value.shape[-1] != feature_size:
            raise ValueError(
                f"{name} 的 shape 應為 ({feature_size},) 或 "
                f"(batch, {feature_size})，目前為 {tuple(value.shape)}。"
            )

        return value

    def _align_batch_sizes(self, *tensors):
        """功能：對齊多個網路輸入的 batch；單筆資料可展開至共同 batch 大小。"""
        target_batch_size = max(tensor.shape[0] for tensor in tensors)
        aligned_tensors = []

        for tensor in tensors:
            if tensor.shape[0] == target_batch_size:
                aligned_tensors.append(tensor)
            elif tensor.shape[0] == 1:
                aligned_tensors.append(
                    tensor.expand(target_batch_size, -1)
                )
            else:
                raise ValueError(
                    "所有輸入的 batch size 必須相同；只有 batch size=1 "
                    "可以自動展開。"
                )

        return tuple(aligned_tensors)

    def hard_update(self, local_model, target_model):
        """功能：完整複製網路參數，用於 target network 初始化或高層週期更新。"""
        target_model.load_state_dict(
            local_model.state_dict()
        )

    def soft_update(self, local_model, target_model):
        """功能：以 tau 混合參數，更新低層 SAC target Critic。"""
        with torch.no_grad():
            for target_parameter, local_parameter in zip(
                target_model.parameters(),
                local_model.parameters(),
            ):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(
                    local_parameter,
                    alpha=self.tau,
                )

    def update_target_networks(self, update_high=False):
        """功能：更新低層 target Critic，並在指定週期同步 high-level target DQN。"""
        self.soft_update(
            self.critic_network_1,
            self.target_critic_network_1,
        )
        self.soft_update(
            self.critic_network_2,
            self.target_critic_network_2,
        )

        if update_high:
            self.hard_update(
                self.high_network,
                self.target_high_network,
            )

    def set_train_mode(self):
        """功能：將所有 online network 切換為訓練模式。"""
        self.high_network.train()
        self.move_actor.train()
        self.service_actor.train()
        self.critic_network_1.train()
        self.critic_network_2.train()

        # Target networks 不參與梯度更新，固定保持評估模式。
        self.target_high_network.eval()
        self.target_critic_network_1.eval()
        self.target_critic_network_2.eval()

    def set_eval_mode(self):
        """功能：將所有 online network 切換為評估模式。"""
        self.high_network.eval()
        self.target_high_network.eval()
        self.move_actor.eval()
        self.service_actor.eval()
        self.critic_network_1.eval()
        self.critic_network_2.eval()
        self.target_critic_network_1.eval()
        self.target_critic_network_2.eval()

    def get_parameter(self):
        """功能：回傳三層網路與訓練排程的主要超參數，供紀錄實驗設定。"""
        return {
            "model": "agent_Onliine_HSAC",
            "scope": "one_agent_per_uav",
            "device": str(self.device),
            "state_size": self.state_size,
            "num_devices": self.num_devices,
            "num_uavs": self.num_uavs,
            "goal_size": self.goal_size,
            "move_action_size": self.move_action_size,
            "service_action_size": self.service_action_size,
            "move_input_size": self.move_input_size,
            "service_input_size": self.service_input_size,
            "critic_input_size": self.critic_input_size,
            "critic_state_contract": "moved_state",
            "hidden_size": self.hidden_size,
            "high_interval": self.high_interval,
            "max_movement": self.max_movement,
            "high_gamma": self.high_gamma,
            "low_gamma": self.low_gamma,
            "tau": self.tau,
            "high_lr": self.high_lr,
            "move_actor_lr": self.move_actor_lr,
            "service_actor_lr": self.service_actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_move": self.alpha_move,
            "alpha_service": self.alpha_service,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "high_target_update_interval": (
                self.high_target_update_interval
            ),
            "gradient_clip_norm": self.gradient_clip_norm,
            "has_move_preview_fn": self.move_preview_fn is not None,
        }
