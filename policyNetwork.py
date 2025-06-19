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
        self.fc2 = nn.Linear(hidden_dim, 1)  # scalar output score

    def forward(self, state_move):
        x = torch.relu(self.fc1(state_move))
        score = self.fc2(x)
        return score


class SimpleTransformer(nn.Module):  # try pretrained transformer
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6).to(device)
        self.fc = nn.Linear(960, 1).to(device)

    def forward(self, state_move):
        x = self.transformer_encoder(state_move)
        x = self.fc(x)
        return x

    def calculate_scores(self, state_vec, legal_moves):
        state_vec = state_vec.to(device)  # shape: (832,) if your move-encoding is 128
        # Build a batch of [state||move] vectors
        move_tensors = []
        for move in legal_moves:
            move_vec = encode_move(move).to(device)  # shape: (128,)
            move_tensors.append(torch.cat([state_vec, move_vec], dim=-1))
        # Stack into shape (N, 960) then add sequence dim → (N, 1, 960)
        batch = torch.stack(move_tensors, dim=0).unsqueeze(1)
        scores = self.forward(batch)  # → (N, 1, 1)
        # Flatten and return as Python floats
        return scores.view(-1)
