import numpy as np
import random
import itertools
import math

def generate_unique_coords(N):
    """
    生成 N 個不重複的整數座標，範圍在 [0,0] ~ [9,9]
    
    回傳：
        np.array shape = (N, 2)
    """
    assert 0 <= N <= 100, "N 不能超過 100（因為總共只有 10x10 個點）"
    
    # 建立所有可能座標 (100 個)
    all_coords = np.array([(x, y) for x in range(10) for y in range(10)])
    
    # 隨機選 N 個（不重複）
    indices = np.random.choice(len(all_coords), size=N, replace=False)
    
    return all_coords[indices]


class Environment():
    CHANNEL_GAIN = 10**(-30/10) # 通道增益(-30 dB)
    H = 100 # UAV高度
    CELL_DISTANCE = 100 # 單元格之間的距離
    B = 1e6 # 頻寬
    PACKET_SIZE = 5e6 # 封包大小
    SIGMA = (10 ** (-100/10)) * (10e-3) # 雜訊功率 (-100 dBm)

    steps_mov = 2 # 每個時間步的最大位移(論文中steps_mov = 1)
    
    V = np.matrix([[0, 0], # 移動向量(Idle, E, W, N, S)
                   [steps_mov,0],
                   [-steps_mov,0],
                   [0,steps_mov],
                   [0,-steps_mov]])
    
    max_steps = 100 # 最大步數，到達該步數後終止環境
    
    def __init__(self, device_coord, risky_region,config):
        self.device_coord = device_coord
        self.risky_region = risky_region

        self.M = config.M
        self.U = config.U
        self.nCell = config.nCell
        self.energy_weight = config.energy_weight
        self.penalty = config.penalty
        self.prob_of_risk = config.prob_of_risk
        
        self.AOI_max = 100 # 最大AOI限制

        action_select = list(range(0,self.M+1)) # 不選擇/選擇一個Device
        action_move = list(range(0,5)) # 移動動作(Idle, E, W, N, S)
        self.action_space = list(itertools.product(action_select, action_move)) # 動作組合(笛卡兒積)

        self.nAction = len(self.action_space)
        self.nObservation = self.U * 2 + self.M
        
    # 重置環境
    def reset(self):        

        self.device_coord = generate_unique_coords(10)

        # 初始化UAV位置
        self.UAVs_init_coord = np.array([])
        for _ in range(self.U):
            UAV_coord = np.random.randint(self.nCell, size=2) # 隨機初始化UAV位置
            while(any(np.array_equal(r, UAV_coord) for r in self.risky_region)): # 如果存在UAV位於風險區域則重抽
                UAV_coord = np.random.randint(self.nCell, size=2) 
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

    # 更新UAV的位置
    def Update_Location(self, U_loc, V_selected):
        U_loc = U_loc + V_selected

        # 檢查(x,y)是否超過邊界
        for i in [0,1]:
            U_loc[0,i] = max(0, U_loc[0,i])
            U_loc[0,i] = min(self.nCell - 1, U_loc[0,i])
        
        U_loc = np.asarray(U_loc).reshape(-1)
        return U_loc
        
    # 必要的傳輸功率
    def Power_Calc(self, device , U_loc):
        if device < self.M:
            h_dist = self.CELL_DISTANCE * math.dist(U_loc, self.device_coord[device])
            MIN_PWR = (h_dist ** 2 + self.H ** 2) * (2 ** (self.PACKET_SIZE/self.B) - 1) * self.SIGMA / self.CHANNEL_GAIN
        else:
            MIN_PWR = 0
        return MIN_PWR * self.energy_weight
    
    # 獎勵計算
    def Reward_Calc(self, AOI, power):
        reward = 0
        reward = reward - power - (np.sum(AOI)/self.M)
        return reward
    
    # 風險機率  
    def Risk_prob(self, U_loc):
        if(any(np.array_equal(r, U_loc) for r in self.risky_region)):
            risk = self.prob_of_risk
        else:
            risk = 0
        return risk
    
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
            u_loc = self.U_loc[u*2:u*2+2]
            action = np.array(self.action_space[actions[u]])

            # 拆分動作
            self.u_action_select = action[0]
            self.u_action_move = action[1] 
            self.u_V = self.V[action[1]]
            
            if(self.done[u]==0):
                # 更新UAV位置
                u_loc = self.Update_Location(u_loc,self.u_V)
                
                # 計算基本功率
                u_power = self.Power_Calc(self.u_action_select,u_loc)

                # 取得風險機率
                risk_prob = self.Risk_prob(u_loc)
                if risk_prob > 0:
                    self.risk_count[u] = self.risk_count[u] + 1

                    # 取樣以決定是否施加懲罰
                    sample_prob = random.random()
                    if sample_prob < risk_prob:
                        u_power += self.penalty
                
                self.power += u_power
                self.AOI_Reset(self.u_action_select)
            
            # 更新狀態
            U_loc_next[u*2:u*2+2] = u_loc
        
        # 計算整體獎勵與各UAV獎勵
        self.total_reward = self.Reward_Calc(self.AOI, self.power/self.U)
        for u in range(self.U):
            rewards[u] = self.Reward_Calc(self.AOI, self.power)

        # 聚合狀態
        states_next = np.concatenate((np.asarray(U_loc_next).reshape(-1), self.AOI), axis=None)
        
        # 檢查是否終止
        if(self.steps == self.max_steps):
            self.done = np.ones(self.U)
        if(sum(self.done) == self.U):
            self.DONE = 1

        self.steps += 1
        
        return states_next, rewards, self.done
    