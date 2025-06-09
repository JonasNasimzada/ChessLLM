import torch
import torch.nn as nn

from utils.encoding import encode_move


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


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)

    def forward(self, state_move):
        x = self.transformer_encoder(state_move)
        x = self.fc(x)
        return x

    def calculate_scores(self, state_vec, legal_moves):
        scores = []
        for move in legal_moves:
            move_vec = encode_move(move)  # shape: (128,)
            input_tensor = torch.cat([state_vec, move_vec])  # shape: (960,)
            input_tensor = input_tensor.unsqueeze(0)
            score = self.policy_net(input_tensor)
            scores.append(score)
        return scores
