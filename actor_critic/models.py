import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class CNN_Layer(nn.Module):
    def __init__(self):
        super(CNN_Layer, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 512) 
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self, num_actions, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.cnn_layer = CNN_Layer()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        # scale the pixel
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim = 1)
        return value, pi

    def get_tensor(self, x, dtype = torch.float32): 
        return torch.tensor(x, dtype= dtype).to(self.device)
    
    # returns a column vector of action
    def get_action(self, obs):
        x = self.get_tensor(obs)
        with torch.no_grad():
            _, pi = self(x)
        
        dist = Categorical(pi)
        action = dist.sample().unsqueeze(1).cpu().numpy() 
        return action
    
    # returns a column vector of value
    def get_value(self, obs):
        x = self.get_tensor(obs)
        with torch.no_grad():
            v, _= self(x)
        return v.cpu().numpy()
    
    # takes a column vec of action 
    # returns a column vector of log_prob
    def evaluate_actions(self, pi, action):
        dist = Categorical(pi)
        # print(action.squeeze(-1).size() ) # 30, 1
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        # print( log_prob.size())
        entropy = dist.entropy().mean()
        return log_prob, entropy 