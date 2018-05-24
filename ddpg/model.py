import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fcs = nn.Linear(nb_states, hidden1 // 2)
        self.fca = nn.Linear(nb_actions, hidden1 // 2)
        self.fc1 = nn.Linear(hidden1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        state, action = x
        s = self.fcs(state)
        s = self.relu(s)
        a = self.fca(action)
        a = self.relu(a)
        
        out = self.fc1(torch.cat([s, a], 1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
