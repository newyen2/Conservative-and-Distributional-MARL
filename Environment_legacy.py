import numpy as np
import random
import itertools
import math

def generate_unique_coords(N, min_dist = 0.0, L_corner = (0.0, 0.0), H_corner = (10.0, 10.0), max_attempts=100000):
    assert N >= 0, f"Violate (N >= 0), {N}"
    assert min_dist >= 0, f"Violate (min_dist >= 0), {min_dist}"

    L = np.array(L_corner, dtype=np.float32)
    H = np.array(H_corner, dtype=np.float32)

    assert L.shape == (2,), f"Violate (L.shape == (2,)), {L.shape}"
    assert H.shape == (2,), f"Violate (H.shape == (2,)), {H.shape}"
    assert np.all(L < H), f"Violate (np.all(L < H)), {L}, {H}"

    coords = []
    attempts = 0 # 總嘗試次數

    while len(coords) < N and attempts < max_attempts:
        attempts += 1

        candidate = np.random.uniform(low=L, high=H, size=2)

        if len(coords) == 0:
            coords.append(candidate)
            continue

        distances = np.linalg.norm(np.array(coords) - candidate, axis=1)

        if np.all(distances >= min_dist):
            coords.append(candidate)

    assert len(coords) == N, f"Violate (len(coords) == N), {len(coords)}"
    
    return np.array(coords)

class Environment():
    CHANNEL_GAIN = 10**(-30/10) # 通道增益(-30 dB)
    H = 100 # UAV高度
    CELL_DISTANCE = 100 # 單元格之間的距離
    B = 1e6 # 頻寬
    PACKET_SIZE = 5e6 # 封包大小
    SIGMA = (10 ** (-100/10)) * (10e-3) # 雜訊功率 (-100 dBm)
    
    max_steps = 100 # 最大步數，到達該步數後終止環境
    
    def __init__(self, device_coord, risky_region, config):
        # TODO(HRL): 階層式版本需在此集中保存移動上限與狀態維度契約，避免 Agent 和環境各自推算。
        self.device_coord = device_coord
        self.risky_region = risky_region

        self.M = config.M
        self.U = config.U
        self.nCell = config.nCell
        self.energy_weight = config.energy_weight
        self.penalty = config.penalty
        self.prob_of_risk = config.prob_of_risk
        self.boundary_penalty_weight = config.boundary_penalty_weight
        
        self.AOI_max = 100 # 最大AOI限制

        # 地圖範圍
        self.L_map = np.array([0.0, 0.0], dtype=np.float32)
        self.H_map = np.array([10.0, 10.0], dtype=np.float32)

        self.nAction_move = 2
        self.nAction_select = self.M + 1
        self.nObservation = self.U * 2 + self.M * 3
        
    # 重置環境
    def reset(self):
        # TODO(HRL): 階層式版本應改由 build_state 統一組合 S，確保 S、S' 與 next_state 排列一致。

        self.device_coord = generate_unique_coords(N = self.M)
        # self.device_coord = np.array([[3,1.5],[7,2.5],[6.5,7],[1,6.5],[7.5,5],[8.5,5],[9.5,1],[6.5,1],[4,7.5],[2.5,3]])

        # 初始化UAV位置
        self.UAVs_init_coord = np.array([])

        for _ in range(self.U):
            UAV_coord = np.random.uniform(
                low=self.L_map,
                high=self.H_map,
                size=2
            ).astype(np.float32)

            while self.is_coord_risky(UAV_coord):
                UAV_coord = np.random.uniform(
                    low=self.L_map,
                    high=self.H_map,
                    size=2
                ).astype(np.float32)

            self.UAVs_init_coord = np.append(self.UAVs_init_coord, UAV_coord)
        
        self.AOI = np.ones(self.M) # 初始化AoI

        self.steps = 1

        self.done = np.zeros(self.U) # 個別UAV是否已完成
        self.DONE = 0 # 整體環境是否終止

        self.risk_count = np.zeros(self.U) # UAV進入風險區域的次數 
            
        # return np.concatenate((np.asarray(self.UAVs_init_coord).reshape(-1), self.AOI), axis=None)
        return np.concatenate((np.asarray(self.UAVs_init_coord).reshape(-1), self.AOI, np.asarray(self.device_coord).reshape(-1)), axis=None)

    def build_state(self, uav_locations, aoi):
        """依固定順序建立環境狀態向量。

        產生理由：集中定義 state、移動後狀態 S' 與 next_state 的欄位排列，
        避免高層、低層及 replay buffer 各自拼接而產生格式差異。

        狀態格式為：UAV 座標 (2U)、AOI (M)、裝置座標 (2M)。
        此方法只建立新陣列，不修改 Environment 內部狀態。
        """
        uav_locations = np.asarray(uav_locations, dtype=np.float32).reshape(-1)
        aoi = np.asarray(aoi, dtype=np.float32).reshape(-1)
        device_locations = np.asarray(
            self.device_coord,
            dtype=np.float32,
        ).reshape(-1)

        expected_uav_size = self.U * 2
        expected_aoi_size = self.M
        expected_device_size = self.M * 2

        if uav_locations.size != expected_uav_size:
            raise ValueError(
                "uav_locations must contain "
                f"{expected_uav_size} values, got {uav_locations.size}."
            )
        if aoi.size != expected_aoi_size:
            raise ValueError(
                f"aoi must contain {expected_aoi_size} values, got {aoi.size}."
            )
        if device_locations.size != expected_device_size:
            raise ValueError(
                "device_coord must contain "
                f"{expected_device_size} values, got {device_locations.size}."
            )

        state = np.concatenate(
            (uav_locations, aoi, device_locations),
            axis=None,
        ).astype(np.float32, copy=False)

        if state.size != self.nObservation:
            raise RuntimeError(
                "Built state size does not match nObservation: "
                f"{state.size} != {self.nObservation}."
            )

        return state

    def split_state(self, state):
        """新增理由：集中解析 UAV 座標、AOI 與裝置座標，避免各模組重複硬編碼切片位置。"""
        pass

    def preview_move(self, states, move_actions):
        """新增理由：SelectActor 必須先取得移動後狀態 S'，且預覽階段不可提前改變環境內部狀態。"""
        pass

    def get_service_mask(self, moved_state):
        """新增理由：根據 S' 統一產生可服務動作遮罩，並保留 M+1 的不服務選項。"""
        pass
    
    # 重置AOI
    def AOI_Reset(self, device):
        if device < self.M:
            self.AOI[device] = 1

    def is_coord_risky(self, coord):
        x, y = coord

        for region in self.risky_region:
            xmin, ymin, xmax, ymax = region

            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True

        return False

    # 更新UAV的位置
    # def Update_Location(self, U_loc, V_selected):
    #     U_loc = U_loc + V_selected

    #     # 檢查(x,y)是否超過邊界
    #     U_loc = np.clip(
    #         U_loc,
    #         self.L_map,
    #         self.H_map
    #     )

    #     U_loc = np.asarray(U_loc).reshape(-1)
    #     return U_loc
    def Update_Location(self, U_loc, target_loc):
        # TODO(HRL): 目前輸入是絕對目標座標；階層式 MoveActor 將輸出受 max_mov 限制的位移或方向。
        move_vec = target_loc - U_loc

        move_dist = np.linalg.norm(move_vec)
        
        if move_dist > 2:
            move_vec = move_vec / move_dist * 2

        raw_next_loc = U_loc + move_vec


        clipped_next_loc = np.clip(
            raw_next_loc,
            self.L_map,
            self.H_map
        )

        boundary_violation = np.linalg.norm(raw_next_loc - clipped_next_loc)

        clipped_next_loc = np.asarray(clipped_next_loc).reshape(-1)

        return clipped_next_loc, boundary_violation
        
    # 必要的傳輸功率
    def Power_Calc(self, device , U_loc):
        if device < self.M:
            h_dist = self.CELL_DISTANCE * math.dist(U_loc, self.device_coord[device])
            MIN_PWR = (h_dist ** 2 + self.H ** 2) * (2 ** (self.PACKET_SIZE/self.B) - 1) * self.SIGMA / self.CHANNEL_GAIN
        else:
            MIN_PWR = 0
        return MIN_PWR
    
    # 獎勵計算
    def Reward_Calc(self, AOI, power, boundary_penalty):
        reward = 0
        reward = reward - power - np.sum(AOI)/self.M - boundary_penalty
        return reward
    
    # 風險機率  
    def Risk_prob(self, U_loc):
        x, y = U_loc

        if self.is_coord_risky(U_loc):
            return self.prob_of_risk
        else:
            return 0
    
    # 執行環境
    def step(self, states, actions):
        # TODO(HRL): 此為舊 Hybrid SAC 的合併動作入口；新流程需使用 step_hierarchical 分離移動與服務時序。
        
        # 拆分狀態
        self.U_loc = states[0 : self.U*2]
        self.AOI = states[self.U*2 : self.U*2 + self.M]
        
        # AOI自然增長
        self.AOI = np.minimum(self.AOI + 1, self.AOI_max)

        self.power = 0
        U_loc_next = np.zeros(self.U*2)
        rewards = [0]*self.U

        total_boundary_penalty = 0
        
        for u in range(self.U):
            # 取得UAV當前位置與動作
            u_loc = self.U_loc[u*2 : u*2+2]
            action = actions[u]

            # 拆分動作
            self.u_action_select = action[0]
            self.AOI_Reset(self.u_action_select)

        for u in range(self.U):
            # 取得UAV當前位置與動作
            u_loc = self.U_loc[u*2 : u*2+2]
            action = actions[u]

            # 拆分動作
            self.u_action_select = action[0]
            self.u_action_move = action[1]
            
            if(self.done[u]==0):
                # 更新UAV位置
                u_loc, boundary_violation = self.Update_Location(u_loc,self.u_action_move)

                # 計算基本功率
                u_power = self.Power_Calc(self.u_action_select,u_loc) * self.energy_weight

                boundary_penalty = self.boundary_penalty_weight * boundary_violation
                total_boundary_penalty += boundary_penalty

                # 取得風險機率
                risk_prob = self.Risk_prob(u_loc)
                if risk_prob > 0:
                    self.risk_count[u] = self.risk_count[u] + 1

                    # 取樣以決定是否施加懲罰
                    sample_prob = random.random()
                    if sample_prob < risk_prob:
                        u_power += self.penalty

                rewards[u] = self.Reward_Calc(self.AOI, u_power, boundary_penalty)

                self.power += u_power
            
            # 更新狀態
            U_loc_next[u*2 : u*2+2] = u_loc
        
        # 計算整體獎勵與各UAV獎勵
        total_reward = self.Reward_Calc(self.AOI, self.power/self.U, total_boundary_penalty/self.U)
        for u in range(self.U):
            rewards[u] = total_reward

        self.total_reward = self.Reward_Calc(self.AOI, self.power/self.U, 0)

        # 聚合狀態
        # states_next = np.concatenate((np.asarray(U_loc_next).reshape(-1), self.AOI), axis=None)
        states_next = np.concatenate((np.asarray(U_loc_next).reshape(-1), self.AOI, np.asarray(self.device_coord).reshape(-1)), axis=None)
        
        # 檢查是否終止
        if(self.steps == self.max_steps):
            self.done = np.ones(self.U)
        if(sum(self.done) == self.U):
            self.DONE = 1

        self.steps += 1
        
        return states_next, rewards, self.done

    def step_hierarchical(self, states, move_actions, service_actions, move_preview=None):
        """新增理由：依序提交 move、S'、service 並計算獎勵，同時保留舊 step 介面供既有程式使用。"""
        pass
