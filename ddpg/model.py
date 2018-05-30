import numpy as np
import torch
import torch.nn as nn


def initialize(size):
    fanin = size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.parameters_initialization(init)

    def parameters_initialization(self, init):
        self.fc1.weight.data = initialize(self.fc1.weight.data.size())
        self.fc2.weight.data = initialize(self.fc2.weight.data.size())
        nn.init.uniform_(self.fc3.weight.data, -init, init)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.parameters_initialization(init)

    def parameters_initialization(self, init):
        self.fc1.weight.data = initialize(self.fc1.weight.data.size())
        self.fc2.weight.data = initialize(self.fc2.weight.data.size())
        nn.init.uniform_(self.fc3.weight.data, -init, init)

    def forward(self, x):
        state, action = x
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, action], 1))
        out = self.relu(out)
        out = self.fc3(out)

        return out
