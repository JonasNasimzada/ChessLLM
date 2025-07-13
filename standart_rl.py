import tkinter as tk
import chess
import time
import threading
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from policyNetwork import SimpleTransformer

#############################################
# State and Action Encoding Functions
#############################################

# Mapping for pieces:
# Use index 0 for empty, then 1-6 for White pieces and 7-12 for Black pieces.
piece_to_index = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_board(board):
    """
    Encode the board as a flattened one-hot tensor.
    Each of 64 squares is represented as a one-hot vector of length 13.
    (Index 0 indicates an empty square.)
    Returns a torch tensor of shape (832,).
    """
    encoding = torch.zeros(64, 13)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.symbol()]
            encoding[square, idx] = 1.0
        else:
            encoding[square, 0] = 1.0
    return encoding.flatten()


def encode_move(move):
    """
    Encode a move as the concatenation of two one-hot vectors for the source and target squares.
    Each one-hot is length 64, so the result is a 128-dimension tensor.
    """
    move_encoding = torch.zeros(128)
    move_encoding[move.from_square] = 1.0
    move_encoding[64 + move.to_square] = 1.0
    return move_encoding


#############################################
# Policy Network (Transformer Agent)
#############################################

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # scalar output score

    def forward(self, state_move):
        x = torch.relu(self.fc1(state_move))
        score = self.fc2(x)
        return score


#############################################
# RL Agent Definition (Combined Loss per Episode)
#############################################

class RLAgent:
    def __init__(self, lr=1e-3, rl_model=SimpleTransformer):
        self.policy_net = rl_model
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # These lists store the log probabilities for moves during the episode.
        self.episode_log_probs = []
        # A variable to accumulate loss (from immediate rewards) over the episode.
        self.cumulative_loss = 0.0

    def choose_move(self, board):
        """
        Compute scores for each legal move (using board+move encodings),
        sample from the softmax distribution, and save the log probability.
        The log probability is stored for use in the episode-level bonus loss.
        Returns the chosen move.
        """
        state_vec = encode_board(board)  # shape: (832,)
        state_vec = state_vec.to(device)
        legal_moves = list(board.legal_moves)
        scores = []
        for move in legal_moves:
            move_vec = encode_move(move).to(device)  # shape: (128,)
            input_tensor = torch.cat([state_vec, move_vec])  # shape: (960,)
            input_tensor = input_tensor.unsqueeze(0)
            score = self.policy_net(input_tensor)
            scores.append(score)
        scores_tensor = torch.stack(scores).view(-1)  # ensure one-dimensional tensor, shape: (n_moves,)
        probs = torch.softmax(scores_tensor, dim=0)
        m = torch.distributions.Categorical(probs)
        index = m.sample()
        log_prob = m.log_prob(index)
        self.episode_log_probs.append(log_prob)
        chosen_move = legal_moves[index.item()]
        return chosen_move

    def accumulate_immediate_loss(self, immediate_reward):
        """
        Accumulate the immediate loss for the most recent move.
        immediate_loss = - log_prob * immediate_reward.
        Instead of updating immediately, add this to cumulative_loss.
        """
        if self.episode_log_probs:
            log_prob = self.episode_log_probs[-1]
            self.cumulative_loss += -log_prob * immediate_reward

    def finalize_episode(self, final_reward, weight=10):
        """
        Compute bonus loss, apply backprop, step optimizer,
        return total loss for logging.
        """
        if self.episode_log_probs:
            for log_prob in self.episode_log_probs:
                self.cumulative_loss += -log_prob * final_reward * weight

        # backprop & step
        self.optimizer.zero_grad()
        self.cumulative_loss.backward()
        self.optimizer.step()

        # grab scalar loss for logging
        loss_value = self.cumulative_loss.item() if isinstance(self.cumulative_loss,
                                                               torch.Tensor) else self.cumulative_loss

        # reset for next episode
        self.episode_log_probs = []
        self.cumulative_loss = 0.0

        return loss_value

    def save_checkpoint(self, filename="policy_checkpoint.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print("Checkpoint saved.")

    def load_checkpoint(self, filename="policy_checkpoint.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(filename))
            print("Checkpoint loaded.")
        except Exception as e:
            print("Could not load checkpoint:", e)

    def save_checkpoint(self, filename="policy_checkpoint.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print("Checkpoint saved.")

    def load_checkpoint(self, filename="policy_checkpoint.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(filename))
            print("Checkpoint loaded.")
        except Exception as e:
            print("Could not load checkpoint:", e)


#############################################
# Classical Agent (Minimax with Alpha-Beta)
#############################################

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def evaluate_material(board):
    """A simple material evaluation function."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color else -value
    return score


def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_material(board)
    if maximizing:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval


def classical_agent_move(board, depth=3):
    legal_moves = list(board.legal_moves)
    best_moves = []
    best_eval = -float('inf')
    for move in legal_moves:
        board.push(move)
        current_eval = minimax(board, depth - 1, -float('inf'), float('inf'), False)
        board.pop()
        if current_eval > best_eval:
            best_eval = current_eval
            best_moves = [move]
        elif current_eval == best_eval:
            best_moves.append(move)
    if not best_moves:
        best_moves = legal_moves
    chosen_move = random.choice(best_moves)
    return chosen_move


#############################################
# GUI Application with Continual RL & Long-Term Rewards
#############################################

# Unicode pieces for drawing the board.
piece_unicode = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}


class ChessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformer (RL) vs Classical Chess with Two-Phase Rewards")
        self.board = chess.Board()
        self.square_size = 80
        self.canvas = tk.Canvas(self, width=self.square_size * 8,
                                height=self.square_size * 8)
        self.canvas.pack()
        self.status_label = tk.Label(self, text="Game start", font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=10)
        self.last_move = None
        self.fast_mode = True  # For first 100 games: fast mode (no UI, no delays)
        self.game_count = 0
        self.withdraw()  # Hide the window in fast mode.
        threading.Thread(target=self.game_loop, daemon=True).start()

    def draw_board(self):
        self.canvas.delete("all")
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = "bisque" if (file + rank) % 2 == 0 else "sienna"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece_unicode[piece.symbol()]
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                            text=symbol, font=("Arial", 36))
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=3)

    def game_loop(self):
        # checkpoint_file = "policy_checkpoint_Transformer_pretrained.pth"
        while self.game_count < 100:
            self.board.reset()
            self.last_move = None
            self.game_count += 1
            print(f"Starting game {self.game_count}")
            self.status_label.config(text=f"Game {self.game_count} start")

            # Clear accumulated losses for a new episode.
            rl_agent.episode_log_probs = []
            rl_agent.cumulative_loss = 0.0

            # Play one complete game.
            while not self.board.is_game_over():
                if self.board.turn:  # RL (White)
                    # old_eval = evaluate_material(self.board)
                    move = rl_agent.choose_move(self.board)
                    self.board.push(move)
                    # new_eval = evaluate_material(self.board)
                    # immediate_reward = new_eval - old_eval

                    # Accumulate and log immediate loss
                    # rl_agent.accumulate_immediate_loss(immediate_reward)
                    wandb.log({
                        # "immediate_reward": immediate_reward,
                        "move_number": len(rl_agent.episode_log_probs),
                        "episode": self.game_count
                    })
                else:  # Classical (Black)
                    move = classical_agent_move(self.board, depth=wandb.config.minimax_depth)
                    self.board.push(move)

            # End of game.
            result = self.board.result()
            if result == "1-0":
                final_reward = 10
            elif result == "0-1":
                final_reward = -10
            else:
                final_reward = -10

            # Finalize episode, get loss for logging
            # episode_loss = rl_agent.finalize_episode(final_reward, weight=wandb.config.reward_weight)

            # Log episode metrics
            wandb.log({
                "episode": self.game_count,
                # "episode_loss": episode_loss,
                "final_reward": final_reward,
                "result": result
            })

            print(f"Game {self.game_count} over: {result}")
            # rl_agent.save_checkpoint(filename=checkpoint_file)
        self.quit()


class LinearLayer(nn.Module):  # try pretrained transformer
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.model(x)
        return self.fc1(x)


if __name__ == "__main__":
    wandb.init(
        project="chess-rl",
        config={
            "learning_rate": 1e-3,
            "reward_weight": 10,
            "minimax_depth": 3
        }
    )

    # Instantiate or load agent
    # checkpoint_file = "policy_checkpoint.pth"
    # if os.path.exists(checkpoint_file):
    #     rl_agent = RLAgent(lr=wandb.config.learning_rate)
    #     rl_agent.load_checkpoint(checkpoint_file)
    # else:
    # rl_agent = RLAgent(lr=wandb.config.learning_rate)
    # rl_agent = RLAgent(lr=wandb.config.learning_rate)

    model = SimpleTransformer().to(device)
    # model.load_state_dict(torch.load("checkpoints/pretrain_transformer/v2/epoch680.pt",))
    model = LinearLayer(model).to(device)
    model.load_state_dict(torch.load("policy_checkpoint_Transformer_pretrained.pth", map_location=device))
    rl_agent = RLAgent(lr=wandb.config.learning_rate, rl_model=model)

    # Track gradients & parameters
    wandb.watch(rl_agent.policy_net, log="all")

    # Start GUI/training
    app = ChessApp()
    app.mainloop()
