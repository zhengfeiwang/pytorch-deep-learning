import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fc(torch.cat([hidden[0], hidden[1]], 1))

        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)
