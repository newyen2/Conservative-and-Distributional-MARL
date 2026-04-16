import pandas as pd
from utils import get_config
import matplotlib.pyplot as plt


config = get_config()

# sub = ['CIQR-CVaR','CCQR-CVaR','CIQR','CCQR','CIQL','CCQL','DIQN','DCQN','QR-DIQN','QR-DCQN']
sub = ['CIQR-CVaR','CCQR-CVaR','CIQR','CCQR','CIQL','CCQL']
# setting = [("#01A8A8",'-','o'),('#01A8A8',':','o'),
#            ('blue','-','x'),('blue',':','x'),
#            ('red','-','>'),('red',':','>'),
#            ('black','-','P'),('black',':','P'),
#            ('green','-','.'),('green',':','.')]
setting = [("#01A8A8",'-','o'),('#01A8A8',':','o'),
           ('blue','-','x'),('blue',':','x'),
           ('red','-','>'),('red',':','>')]
result = []
for i in sub:
    rf = (pd.read_csv(config.PATH+fr'\Result_Datas\16%_2UAVs_300pen_seed1\Result_Offline_{i}_16%_2UAVs_300pen.csv').iloc[:, 0] / 1000).tolist()
    r = (pd.read_csv(config.PATH+f'\Result_Datas\Result_Offline_{i}_16%_2UAVs_300pen.csv').iloc[:, 0] / 1000).tolist()
    
    result.append([a - b for a, b in zip(r, rf)])


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
           bbox_to_anchor=(0.5, 0),
           ncol=2)
plt.grid(True)
# plt.savefig(config.PATH+r'\Results\\Total_test_reward.png', bbox_inches='tight')
plt.show()


