import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class SAC_Continuous_Actor(nn.Module):
    def __init__(self, input_size, action_size, layer_size):
        """建立低層移動策略網路。

        產生理由：階層式架構將連續移動與離散服務拆開，因此 MoveActor
        只負責輸出連續動作分布的平均值與 log standard deviation。
        輸入維度由 Agent 決定，Network 不解析 state 或 goal 的內容。
        """
        super(SAC_Continuous_Actor, self).__init__()

        self.input_size = input_size
        self.action_size = action_size

        self.head_1 = nn.Linear(input_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.continuous_mu = nn.Linear(layer_size, action_size)
        self.continuous_log_std = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """回傳連續動作分布參數，不在 Network 內進行取樣或範圍縮放。

        產生理由：log_std clamp、reparameterization、tanh 與 max_mov 縮放
        屬於策略取樣流程，應由 Agent 統一處理。
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))

        continuous_mu = self.continuous_mu(x)
        continuous_log_std = self.continuous_log_std(x)

        return continuous_mu, continuous_log_std


class SAC_Discrete_Actor(nn.Module):
    def __init__(self, input_size, action_size, layer_size):
        """建立低層服務選擇策略網路。

        產生理由：SelectActor 使用移動後狀態 S' 與 high-level goal，輸出
        M+1 個服務動作 logits；輸入與輸出維度仍由 Agent 傳入。
        """
        super(SAC_Discrete_Actor, self).__init__()

        self.input_size = input_size
        self.action_size = action_size

        self.head_1 = nn.Linear(input_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.discrete_logits = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """回傳未正規化的服務 logits。

        產生理由：合法動作 mask、softmax 與 categorical sampling 需要環境及
        策略資訊，因此留給 Agent 處理，Network 只提供最基本的映射。
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))

        discrete_logits = self.discrete_logits(x)

        return discrete_logits


class SAC_Critic(nn.Module):
    def __init__(self, input_size, layer_size):
        """建立評估完整階層式低層動作的 joint Critic。

        產生理由：MoveActor 與 SelectActor 必須由同一個 Q-value 協同更新；
        state、goal、move 與 service 的拼接方式由 Agent 負責。
        """
        super(SAC_Critic, self).__init__()

        self.input_size = input_size

        self.head_1 = nn.Linear(input_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)

    def forward(self, input):
        """接收 Agent 組合完成的 joint input，為每筆資料輸出一個 Q-value。

        產生理由：Critic 不自行切割 tensor，可避免 Network 與狀態定義、
        goal 編碼方式或服務動作數量耦合。
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        q_value = self.ff_2(x)

        return q_value
