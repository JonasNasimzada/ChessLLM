import torch
import torch.nn as nn

from utils.encoding import encode_move

#############################################
# Policy Network (Transformer Agent)
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearNetwork(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.

    Args:
        input_dim (int): The size of the input layer. Default is 960.
        hidden_dim (int): The size of the hidden layer. Default is 256.
    """

    def __init__(self, input_dim=960, hidden_dim=256):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer

    def forward(self, state_move):
        """
        Forward pass through the network.

        Args:
            state_move (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = torch.relu(self.fc1(state_move))  # Apply ReLU activation
        score = self.fc2(x)  # Compute the final score
        return score


class SimpleTransformer(nn.Module):
    """
    A transformer-based model for processing state and move representations.

    This model uses a TransformerEncoder with 6 layers and outputs a single score.

    Args:
        None
    """

    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)  # Fully connected layer for final output

    def forward(self, state_move):
        """
        Forward pass through the transformer.

        Args:
            state_move (torch.Tensor): Input tensor of shape (batch_size, 960) or (batch_size, 1, 960).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if state_move.dim() == 2:
            state_move = state_move.unsqueeze(1)  # Add a batch dimension if missing
        x = self.transformer_encoder(state_move)  # Pass through the transformer encoder
        x = x.squeeze(1)  # Remove the extra dimension
        return self.fc(x)  # Compute the final score

    def calculate_scores(self, state_vec, legal_moves):
        """
        Calculate scores for a batch of legal moves.

        Args:
            state_vec (torch.Tensor): Encoded state vector of shape (960,).
            legal_moves (list): List of legal moves.

        Returns:
            torch.Tensor: Scores for each move, shape (num_moves,).
        """
        state_vec = state_vec.to(device)
        move_tensors = []
        for move in legal_moves:
            move_vec = encode_move(move).to(device)  # Encode each move
            move_tensors.append(torch.cat([state_vec, move_vec], dim=-1))  # Concatenate state and move
        batch = torch.stack(move_tensors, dim=0).unsqueeze(1)  # Create a batch
        scores = self.forward(batch)  # Compute scores
        return scores.view(-1)  # Flatten the scores


class LinearLayer(nn.Module):
    """
    A wrapper for a pretrained model with an additional linear layer.

    Args:
        pretrained_model (nn.Module): The pretrained model to wrap.
    """

    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model  # Pretrained model
        self.fc1 = nn.Linear(128, 1)  # Additional linear layer

    def forward(self, x):
        """
        Forward pass through the model and the additional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.model(x)  # Pass through the pretrained model
        return self.fc1(x)  # Pass through the additional layer


class PlainSimpleTransformer(nn.Module):
    """
    A plain transformer-based model for processing state and move representations.

    This model uses a TransformerEncoder with 6 layers and outputs a single score.

    Args:
        None
    """

    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(960, 1)  # Fully connected layer for final output

    def forward(self, state_move):
        """
        Forward pass through the transformer.

        Args:
            state_move (torch.Tensor): Input tensor of shape (batch_size, 960) or (batch_size, 1, 960).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if state_move.dim() == 2:
            state_move = state_move.unsqueeze(1)  # Add a batch dimension if missing
        x = self.transformer_encoder(state_move)  # Pass through the transformer encoder
        x = x.squeeze(1)  # Remove the extra dimension
        return self.fc(x)  # Compute the final score
