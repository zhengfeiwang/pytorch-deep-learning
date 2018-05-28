import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def initialize(size, init=None):
    if init:
        return Parameter(torch.Tensor(size).uniform_(-init, init))
    else:
        fan_in = size[0]
        r = 1. / np.sqrt(fan_in)
        return Parameter(torch.Tensor(size).uniform_(-r, r))


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init=3e-3, use_bn=True):
        super(Actor, self).__init__()
        self.use_bn = use_bn
        self.bn1 = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.bn2 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.parameters_initialization(init)

    def parameters_initialization(self, init):
        self.fc1.weight = initialize(self.fc1.weight.size())
        self.fc1.bias = initialize(self.fc1.bias.size())
        self.fc2.weight = initialize(self.fc2.weight.size())
        self.fc2.bias = initialize(self.fc2.bias.size())
        self.fc3.weight = initialize(self.fc3.weight.size(), init=init)
        self.fc3.bias = initialize(self.fc3.bias.size(), init=init)

    def forward(self, x):
        if self.use_bn:
            out = self.bn1(x)
            out = self.fc1(out)
        else:
            out = self.fc1(x)
        out = self.relu(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.fc2(out)
        out = self.relu(out)
        if self.use_bn:
            out = self.bn3(out)
        out = self.fc3(out)
        out = self.tanh(out)

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init=3e-3, use_bn=True):
        super(Critic, self).__init__()
        self.use_bn = use_bn
        self.bns = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.parameters_initialization(init)

    def parameters_initialization(self, init):
        self.fc1.weight = initialize(self.fc1.weight.size())
        self.fc1.bias = initialize(self.fc1.bias.size())
        self.fc2.weight = initialize(self.fc2.weight.size())
        self.fc2.bias = initialize(self.fc2.bias.size())
        self.fc3.weight = initialize(self.fc3.weight.size(), init=init)
        self.fc3.bias = initialize(self.fc3.bias.size(), init=init)

    def forward(self, x):
        state, action = x
        if self.use_bn:
            s = self.bns(state)
            s = self.fc1(s)
        else:
            s = self.fc1(state)
        s = self.relu(s)
        out = self.fc2(torch.cat([s, action], 1))
        out = self.relu(out)
        out = self.fc3(out)

        return out
