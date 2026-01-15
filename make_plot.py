import pandas as pd
from utils import get_config
import matplotlib.pyplot as plt


config = get_config()

sub = ['CIQR-CVaR','CCQR-CVaR','CIQR','CCQR','CIQL','CCQL','DIQN','DCQN','QR-DIQN','QR-DCQN']
setting = [("#01A8A8",'-','o'),('#01A8A8',':','o'),
           ('blue','-','x'),('blue',':','x'),
           ('red','-','>'),('red',':','>'),
           ('black','-','P'),('black',':','P'),
           ('green','-','.'),('green',':','.')]
result = []
for i in sub:
    r = (pd.read_csv(config.PATH+f'Result_Datas\{i}_offline_16_pen_300.csv').iloc[:, 0] / 1000).tolist()
    result.append(r)

plt.figure(figsize=(8, 6))
window_size = 10
for i in range(len(sub)):
    Episode_Reward_smooth = pd.Series(result[i]).rolling(window=window_size).mean()
    Episode_Reward_smooth_10 = Episode_Reward_smooth.iloc[9::10]
    Episode_Reward_smooth_10.index -= 9

    plt.plot(Episode_Reward_smooth_10, label=f'MA-{sub[i]}', color = setting[i][0], 
             linestyle = setting[i][1], marker= setting[i][2], markeredgewidth=2.5)
plt.xlabel('Epochs')
plt.ylabel('Test Return')
plt.legend(loc='lower center',
           bbox_to_anchor=(0.5, 1.02),
           ncol=2)
plt.grid(True)
plt.savefig(config.PATH+r'Results\\Total_test_reward.png', bbox_inches='tight')



