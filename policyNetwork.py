import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Policy Networks
# ---------------------------------------------------------------------

class LinearNetwork(nn.Module):
    """A simple 2‑layer MLP that outputs a scalar score for (state+move)."""

    def __init__(self, input_dim: int = 960, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state_move):
        x = torch.relu(self.fc1(state_move))
        return self.fc2(x)                  # (batch, 1)


class SimpleTransformer(nn.Module):
    """Transformer encoder → **128‑logit** head."""

    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)       # ← was 1, now 128

    def forward(self, state_move):
        if state_move.dim() == 2:
            state_move = state_move.unsqueeze(1)   # (B,1,960)
        x = self.transformer_encoder(state_move)
        x = x.squeeze(1)                           # (B,960)
        return self.fc(x)                          # (B,128)