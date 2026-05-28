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
        self.device_coord = device_coord
        self.risky_region = risky_region

        self.M = config.M
        self.U = config.U
        self.nCell = config.nCell
        self.energy_weight = config.energy_weight
        self.penalty = config.penalty
        self.prob_of_risk = config.prob_of_risk
        self.boundary_penalty_weight = config.boundary_penalty_weight
        self.max_power = config.max_power
        
        self.AOI_max = 100 # 最大AOI限制

        # 地圖範圍
        self.L_map = np.array([0.0, 0.0], dtype=np.float32)
        self.H_map = np.array([10.0, 10.0], dtype=np.float32)

        self.nAction_move = 2
        self.nAction_select = self.M + 1
        self.nObservation = self.U * 2 + self.M
        
    # 重置環境
    def reset(self):        

        # self.device_coord = generate_unique_coords(N = self.M)
        self.device_coord = np.array([[3,1.5],[7,2.5],[6.5,7],[1,6.5],[7.5,5],[8.5,5],[9.5,1],[6.5,1],[4,7.5],[2.5,3]])

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
            
        return np.concatenate((np.asarray(self.UAVs_init_coord).reshape(-1), self.AOI), axis=None)
    
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
    def Update_Location(self, U_loc, V_selected):
        raw_next_loc = U_loc + V_selected

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
        
        # 拆分狀態
        self.U_loc = states[0 : self.U*2]
        self.AOI = states[self.U*2 : self.U*2 + self.M]
        
        # AOI自然增長
        self.AOI = np.minimum(self.AOI + 1, self.AOI_max)

        self.power = 0
        U_loc_next = np.zeros(self.U*2)
        rewards = [0]*self.U
        
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

                if u_power <= self.max_power:
                    self.AOI_Reset(self.u_action_select)    

                boundary_penalty = self.boundary_penalty_weight * boundary_violation

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
        self.total_reward = self.Reward_Calc(self.AOI, self.power/self.U, 0)

        # 聚合狀態
        states_next = np.concatenate((np.asarray(U_loc_next).reshape(-1), self.AOI), axis=None)
        
        # 檢查是否終止
        if(self.steps == self.max_steps):
            self.done = np.ones(self.U)
        if(sum(self.done) == self.U):
            self.DONE = 1

        self.steps += 1
        
        return states_next, rewards, self.done