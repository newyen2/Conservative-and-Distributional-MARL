"""多 UAV 服務任務的階層式 RL 環境。"""

import math

import numpy as np


class Environment:
    """管理環境設定與 episode 狀態。"""

    # 通訊參數：通道增益、高度、距離尺度、頻寬、封包大小與雜訊功率。
    CHANNEL_GAIN = 10 ** (-30 / 10)
    H = 100
    CELL_DISTANCE = 100
    B = 1e6
    PACKET_SIZE = 5e6
    SIGMA = (10 ** (-90 / 10)) * 1e-3

    def __init__(
        self,
        num_devices=10,
        num_uavs=2,
        max_steps=100,
        max_movement=2.0,
        aoi_max=100,
        energy_weight=500.0,
        risk_penalty=300.0,
        risk_probability=0.1,
    ):
        """建立固定環境設定，episode 狀態由後續 reset() 初始化。"""

        # [1. 地圖邊界] 定義二維地圖各軸的最小值與最大值。
        self.map_low = np.array([0.0, 0.0], dtype=np.float32)
        self.map_high = np.array([10.0, 10.0], dtype=np.float32)

        # [2. 任務參數] M 與 U 先定義裝置及 UAV 數量，座標由 reset() 產生。
        self.M = int(num_devices)
        self.U = int(num_uavs)
        self.max_steps = int(max_steps)
        self.max_movement = float(max_movement)
        self.aoi_max = int(aoi_max)
        self.energy_weight = float(energy_weight)
        self.risk_penalty = float(risk_penalty)
        self.risk_probability = float(risk_probability)

        # [3. 環境狀態] reset() 前尚未建立位置、AOI 與 episode 狀態。
        self.device_coord = None
        self.risky_region = None
        self.uav_locations = None
        self.aoi = None
        self.step_count = 0
        self.terminated = False
        self.risk_count = None

    def reset(self):
        """建立固定場景並重置所有 episode 狀態。"""

        # [1. 裝置座標] 每列表示一個裝置的 (x, y) 位置。
        self.device_coord = np.array(
            [
                [3.0, 1.5],
                [7.0, 2.5],
                [6.5, 7.0],
                [1.0, 6.5],
                [7.5, 5.0],
                [8.5, 5.0],
                [9.5, 1.0],
                [6.5, 1.0],
                [4.0, 7.5],
                [2.5, 3.0],
            ],
            dtype=np.float32,
        )

        # [2. 風險區域] 每列表示 (xmin, ymin, xmax, ymax)。
        self.risky_region = np.array(
            [
                [3.0, 2.0, 7.0, 7.0],
            ],
            dtype=np.float32,
        )

        # [3. UAV 位置] 暫時使用一組固定的雙 UAV 散布結果。
        self.uav_locations = np.array(
            [
                [1.5, 8.5],
                [8.5, 1.5],
            ],
            dtype=np.float32,
        )

        # [4. Episode 狀態] AOI 從 1 開始，其餘計數器回到初始值。
        self.aoi = np.ones(self.M, dtype=np.int32)
        self.step_count = 0
        self.terminated = False
        self.risk_count = np.zeros(self.U, dtype=np.int64)

    def AOI_Reset(self, device):
        """將指定裝置的 AOI 重設為 1；M 代表不服務。"""
        if 0 <= device < self.M:
            self.aoi[device] = 1

    def sample_risk(self, coord):
        """判斷座標是否觸發風險懲罰，回傳 0 或 1。"""
        point = np.asarray(coord, dtype=np.float32)

        inside_region = np.any(
            (self.risky_region[:, 0] <= point[0])
            & (point[0] <= self.risky_region[:, 2])
            & (self.risky_region[:, 1] <= point[1])
            & (point[1] <= self.risky_region[:, 3])
        )

        if not inside_region:
            return 0

        return int(np.random.random() < self.risk_probability)

    def Update_Location(self, current_location, move_vector):
        """沿移動向量更新座標，移動距離不超過單步上限。"""
        current_location = np.asarray(current_location, dtype=np.float32)
        move_vector = np.asarray(move_vector, dtype=np.float32)

        move_distance = np.linalg.norm(move_vector)
        if move_distance > self.max_movement:
            move_vector = (
                move_vector / move_distance * self.max_movement
            )

        next_location = current_location + move_vector
        next_location = np.clip(
            next_location,
            self.map_low,
            self.map_high,
        )

        return next_location.astype(np.float32, copy=False)

    def Power_Calc(self, device, U_loc):
        """計算 UAV 服務指定裝置所需的最小發射功率。"""
        if device < self.M:
            h_dist = self.CELL_DISTANCE * math.dist(
                U_loc,
                self.device_coord[device],
            )
            MIN_PWR = (
                (h_dist ** 2 + self.H ** 2)
                * (2 ** (self.PACKET_SIZE / self.B) - 1)
                * self.SIGMA
                / self.CHANNEL_GAIN
            )
        else:
            MIN_PWR = 0

        return MIN_PWR

    def _get_state(self):
        """將目前的 UAV 位置、AOI 與裝置座標整理成一維狀態。"""
        return np.concatenate(
            (
                self.uav_locations.reshape(-1),
                self.aoi,
                self.device_coord.reshape(-1),
            )
        ).astype(np.float32, copy=False)

    def get_state(self):
        """公開回傳目前 observation，避免外部程式直接依賴 private method。"""
        return self._get_state()

    def move_step(self, move_actions):
        """執行所有 UAV 的移動，並回傳服務決策使用的中間狀態。"""
        move_actions = np.asarray(move_actions, dtype=np.float32).reshape(
            self.U,
            2,
        )

        previous_locations = self.uav_locations.copy()
        risk_triggered = np.zeros(self.U, dtype=np.int32)

        # 逐一限制移動距離、更新位置，並取樣新位置的風險事件。
        for uav in range(self.U):
            self.uav_locations[uav] = self.Update_Location(
                self.uav_locations[uav],
                move_actions[uav],
            )
            risk_triggered[uav] = self.sample_risk(
                self.uav_locations[uav]
            )

        self.risk_count += risk_triggered

        # 回傳實際移動量，供外部 reward 設計計算移動成本與風險懲罰。
        move_info = {
            "movement": self.uav_locations - previous_locations,
            "distance": np.linalg.norm(
                self.uav_locations - previous_locations,
                axis=1,
            ).astype(np.float32),
            "risk_triggered": risk_triggered,
        }

        return self._get_state(), move_info

    def service_step(self, service_actions):
        """執行服務、更新 AOI，並完成一個環境 timestep。"""
        provided_actions = np.asarray(
            service_actions,
            dtype=np.int64,
        ).reshape(-1)

        # M 代表不服務；缺少或超出範圍的動作也暫時視為不服務。
        service_targets = np.full(self.U, self.M, dtype=np.int64)
        action_count = min(self.U, provided_actions.size)
        service_targets[:action_count] = provided_actions[:action_count]
        invalid_actions = (
            (service_targets < 0) | (service_targets > self.M)
        )
        service_targets[invalid_actions] = self.M

        # 每完成一個 timestep，所有裝置 AOI 先增加，再重置被服務裝置。
        self.aoi = np.minimum(
            self.aoi + 1,
            self.aoi_max,
        ).astype(np.int32, copy=False)

        power = np.zeros(self.U, dtype=np.float32)
        for uav, device in enumerate(service_targets):
            if device < self.M:
                self.AOI_Reset(int(device))
                power[uav] = self.Power_Calc(
                    int(device),
                    self.uav_locations[uav],
                )

        # 只有服務階段會推進時間並判斷 episode 是否結束。
        self.step_count += 1
        self.terminated = self.step_count >= self.max_steps

        # 保留原始環境量測，reward 由後續模型依自身公式組合。
        service_info = {
            "service_targets": service_targets,
            "power": power,
            "total_power": float(np.sum(power)),
            "mean_aoi": float(np.mean(self.aoi)),
        }

        return self._get_state(), self.terminated, service_info
