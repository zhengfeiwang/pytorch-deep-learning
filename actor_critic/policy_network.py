import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.hidden = nn.Linear(4, 128)
        self.action = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)

        self.rewards = []
        self.saved_actions = []
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        action_score = self.action(x)
        state_value = self.value(x)
        return F.softmax(action_score, dim=-1), state_value
