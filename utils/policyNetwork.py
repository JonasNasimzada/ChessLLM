import torch
import torch.nn as nn

from utils.encoding import encode_move

#############################################
# Policy Network (Transformer Agent)
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearNetwork(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=256):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state_move):
        x = torch.relu(self.fc1(state_move))
        score = self.fc2(x)
        return score


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)

    def forward(self, state_move):
        if state_move.dim() == 2:
            state_move = state_move.unsqueeze(1)  # (B,1,960)
        x = self.transformer_encoder(state_move)
        x = x.squeeze(1)  # (B,960)
        return self.fc(x)

    def calculate_scores(self, state_vec, legal_moves):
        state_vec = state_vec.to(device)
        move_tensors = []
        for move in legal_moves:
            move_vec = encode_move(move).to(device)  # shape: (128,)
            move_tensors.append(torch.cat([state_vec, move_vec], dim=-1))
        batch = torch.stack(move_tensors, dim=0).unsqueeze(1)
        scores = self.forward(batch)  # → (N, 1, 1)
        return scores.view(-1)


class LinearLayer(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.model(x)
        return self.fc1(x)


class PlainSimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)

    def forward(self, state_move):
        if state_move.dim() == 2:
            state_move = state_move.unsqueeze(1)  # (B,1,960)
        x = self.transformer_encoder(state_move)
        x = x.squeeze(1)  # (B,960)
        return self.fc(x)
