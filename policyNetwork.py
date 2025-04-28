import torch
import torch.nn as nn


#############################################
# Policy Network (Transformer Agent)
#############################################

class LinearNetwork(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=256):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # scalar output score

    def forward(self, state_move):
        x = torch.relu(self.fc1(state_move))
        score = self.fc2(x)
        return score
