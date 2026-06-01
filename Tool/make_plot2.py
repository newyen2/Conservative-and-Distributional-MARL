import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import get_config

config = get_config()

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC Beta', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 


# 多個 result_data.csv 的路徑
result_files = {
    r"(8) Independent Reward": config.PATH+fr'\Stored_Datas\8\result_data.csv',
    r"(10) Shared Reward": config.PATH+fr'\Stored_Datas\result_data.csv',
}

window_size = 20
eval_period = 1  # 如果你每 10 episode 評估一次，就填 10；若每 episode 評估一次，就填 1

plt.figure(figsize=(12, 6))

for label, file_path in result_files.items():
    df = pd.read_csv(file_path)

    # 取第一個數值欄位作為 reward
    reward = df.select_dtypes(include="number").iloc[:, 0]

    # rolling average
    reward_smooth = reward.rolling(window=window_size, min_periods=1).mean()

    # x 軸轉成 episode
    x = np.arange(1, len(reward_smooth) + 1) * eval_period

    plt.plot(x, reward_smooth, label=label)

plt.xlabel("Episode")
plt.ylabel("Eval Reward")
plt.title("Comparison of Online SAC Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()