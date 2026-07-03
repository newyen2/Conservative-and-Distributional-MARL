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

class CQR_DQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, N):
        super(CQR_DQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N

        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size*N)

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out.view(input.shape[0], self.N, self.action_size)
    
    def get_action(self,input):
        x = self.forward(input)
        return x.mean(dim=1)
    
class Hybrid_SAC_Actor(nn.Module):
    def __init__(self, state_size, continuous_action_size, discrete_action_size, layer_size):
        super(Hybrid_SAC_Actor, self).__init__()

        self.state_size = state_size
        self.continuous_action_size = continuous_action_size
        self.discrete_action_size = discrete_action_size

        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)

        self.continuous_mu = nn.Linear(layer_size, continuous_action_size)
        self.continuous_log_std = nn.Linear(layer_size, continuous_action_size)

        self.discrete_logits = nn.Linear(layer_size, discrete_action_size)

    def forward(self, state):
        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))

        continuous_mu = self.continuous_mu(x)
        continuous_log_std = self.continuous_log_std(x)

        discrete_logits = self.discrete_logits(x)

        return continuous_mu, continuous_log_std, discrete_logits
    
class Hybrid_SAC_Critic(nn.Module):
    def __init__(self, state_size, continuous_action_size, discrete_action_size, layer_size):
        super(Hybrid_SAC_Critic, self).__init__()

        input_size = state_size + continuous_action_size + discrete_action_size

        self.head_1 = nn.Linear(input_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)

    def forward(self, state, continuous_action, discrete_action_onehot):
        x = torch.cat(
            [state, continuous_action, discrete_action_onehot],
            dim=-1,
        )

        x = torch.relu(self.head_1(x))
        x = torch.relu(self.ff_1(x))
        q_value = self.ff_2(x)

        return q_value