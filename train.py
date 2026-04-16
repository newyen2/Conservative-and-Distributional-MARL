import numpy as np
import time
import torch
from DQN_Online import Train_DQN_Online
from MA_CIQL import Train_MA_CIQL
from MA_CCQL import Train_MA_CCQL
from MA_CIQR import Train_MA_CIQR
from MA_CCQR import Train_MA_CCQR


# 選擇訓練模型
def train(model, device_coord, risky_region):    
    # 由Online DQN生成離線資料集
    if (model=="DQN_Online"):
        Train_DQN_Online(device_coord, risky_region)

    # DQN + CQL
    if(model=="CIQL"):
        Train_MA_CIQL(model, device_coord, risky_region, alpha = 1)
    elif(model=="CCQL"):
        Train_MA_CCQL(model, device_coord, risky_region, alpha = 1)

    # DQN 
    elif(model=="DIQN"):
        Train_MA_CIQL(model, device_coord, risky_region, alpha = 0)
    elif(model=="DCQN"):
        Train_MA_CCQL(model, device_coord, risky_region, alpha = 0)

    # QR-DQN + CQL
    elif(model=="CIQR"):
        Train_MA_CIQR(model, device_coord, risky_region, alpha = 1, eta = 1)
    elif(model=="CCQR"):
        Train_MA_CCQR(model, device_coord, risky_region, alpha = 1, eta = 1)

    # QR-DQN
    elif(model=="QR-DIQN"):
        Train_MA_CIQR(model, device_coord, risky_region, alpha = 0, eta = 1)
    elif(model=="QR-DCQN"):
        Train_MA_CCQR(model, device_coord, risky_region, alpha = 0, eta = 1)

    # QR-DQN + CQL + CVaR
    elif(model=="CIQR-CVaR"):
        Train_MA_CIQR(model, device_coord, risky_region, alpha = 1, eta = 0.15)
    elif(model=="CCQR-CVaR"):
        Train_MA_CCQR(model, device_coord, risky_region, alpha = 1, eta = 0.15)


# 裝置座標
device_coord = np.array([[3,1],[7,2],[6,7],[1,6],[7,5],[8,5],[9,1],[6,1],[4,7],[2,3]])
# device_coord = np.array([[3,1],[7,2],[6,7],[1,6],[7,5],
#                          [8,5],[9,1],[6,1],[4,7],[2,3],
#                          [4,4],[1,9],[9,7],[5,5],[2,8]])


# 危險區域(5*4)
risky_region = np.array([[3,2],[3,3],[3,4],[3,5],[3,6],
                         [4,2],[4,3],[4,4],[4,5],[4,6],
                         [5,2],[5,3],[5,4],[5,5],[5,6],
                         [6,2],[6,3],[6,4],[6,5],[6,6]])

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



model = "CIQR-CVaR"

if __name__ == "__main__":    
    # Wall-time計時
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # np.random.seed(10)
    # device_coord = generate_unique_coords(10)

    train(model, device_coord, risky_region)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Total training wall time: {end_time - start_time:.2f} seconds")
