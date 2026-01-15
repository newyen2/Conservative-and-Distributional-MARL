import numpy as np
import time
import torch
from DQN_Online import Train_DQN_Online
from MA_CIQL import Train_MA_CIQL
from MA_CCQL import Train_MA_CCQL
from MA_CIQR import Train_MA_CIQR
from MA_CCQR import Train_MA_CCQR

# 選擇訓練模型
def train(Model,Dev_Coord,Risky_region):    
    # 由Online DQN生成離線資料集
    if (Model=="DQN_Online"):
        Train_DQN_Online(Dev_Coord,Risky_region)

    # 訓練Offline Agent
    if(Model=="CIQL"): # MA-QL + Conservative + Independent
        alpha = 1
        Train_MA_CIQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="CCQL"): # MA-QL + Conservative + Centralized
        alpha = 1
        Train_MA_CCQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="DIQN"): # MA-DQN + Independent
        alpha = 0
        Train_MA_CIQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="DCQN"): # MA-DQN + Centralized
        alpha = 0
        Train_MA_CCQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="CIQR"): # MA-QR + Conservative + Independent
        alpha = 1
        eta = 1
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CCQR"): # MA-QR + Conservative + Centralized
        alpha = 1
        eta = 1
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="QR-DIQN"): # MA-QR-DQN + Independent
        alpha = 0
        eta = 1
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="QR-DCQN"):  # MA-QR-DQN + Centralized
        alpha = 0
        eta = 1
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CIQR-CVaR"): # MA-QR + Conservative + Independent + CVaR
        alpha = 1
        eta = 0.15
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CCQR-CVaR"): # MA-QR + Conservative + Centralized + CVaR
        alpha = 1
        eta = 0.15
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)

# 裝置座標
Dev_Coord = np.array([[3,1],[7,2],[6,7],[1,6],[7,5],[8,5],[9,1],[6,1],[4,7],[2,3]])

# 危險區域(5*4)
Risky_region = np.array([[3,2],[3,3],[3,4],[3,5],[3,6],
                         [4,2],[4,3],[4,4],[4,5],[4,6],
                         [5,2],[5,3],[5,4],[5,5],[5,6],
                         [6,2],[6,3],[6,4],[6,5],[6,6]])

Model = "DQN_Online"

if __name__ == "__main__":
    
    # Wall-time計時
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    train(Model,Dev_Coord,Risky_region)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Total training wall time: {end_time - start_time:.2f} seconds")
