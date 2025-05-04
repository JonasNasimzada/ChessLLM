import os
import random
import threading
import time
import tkinter as tk

import chess
import torch
import torch.optim as optim

from policyNetwork import LinearNetwork, SimpleTransformer
import rlAgent

# Mapping for pieces:
# Use index 0 for empty, then 1-6 for White pieces and 7-12 for Black pieces.
piece_to_index = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}
# Unicode pieces for drawing the board.
piece_unicode = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}


#############################################
# State and Action Encoding Functions
#############################################


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
# Classical Agent (Minimax with Alpha-Beta)
#############################################


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
        checkpoint_file = "policy_checkpoint.pth"
        while True:
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
                if self.board.turn:  # Transformer (RL Agent) as White.
                    old_eval = evaluate_material(self.board)
                    move = rl_agent.choose_move(self.board)
                    transformer_name = "Transformer (White)"
                    self.last_move = move
                    self.board.push(move)
                    new_eval = evaluate_material(self.board)
                    immediate_reward = new_eval - old_eval
                    # Accumulate immediate loss without updating immediately.
                    rl_agent.accumulate_immediate_loss(immediate_reward)
                else:  # Classical Agent as Black.
                    move = classical_agent_move(self.board, depth=3)
                    classical_name = "Classical (Black)"
                    self.last_move = move
                    self.board.push(move)
                if not self.fast_mode:
                    self.after(0, self.draw_board)
                    # Determine status text based on who moved.
                    status_text = (f"{transformer_name} moved: {move.uci()}"
                                   if self.board.turn else f"{classical_name} moved: {move.uci()}")
                    self.after(0, lambda text=status_text: self.status_label.config(text=text))
                    time.sleep(0.5)
            # End of game.
            result = self.board.result()  # "1-0", "0-1", or "1/2-1/2"
            print(f"Game {self.game_count} over: {result}")
            if result == "1-0":
                final_reward = 10
                transformer_result = "won"
                classical_result = "lost"
            elif result == "0-1":
                final_reward = -10
                transformer_result = "lost"
                classical_result = "won"
            else:
                final_reward = -10
                transformer_result = "drew (punished)"
                classical_result = "drew (punished)"
            # Finalize the episode with bonus update.
            rl_agent.finalize_episode(final_reward, weight=10)
            final_status = (f"Game {self.game_count} over: Transformer (White) {transformer_result}, "
                            f"Classical (Black) {classical_result} (Result: {result})")
            print(final_status)
            self.status_label.config(text=final_status)
            # Save checkpoint to the same file.
            rl_agent.save_checkpoint(filename=checkpoint_file)
            if not self.fast_mode:
                time.sleep(3)
            # After 100 games in fast mode, switch to UI mode.
            if self.game_count == 10 and self.fast_mode:
                print("Completed 100 games in fast mode; switching to UI mode with delays.")
                self.fast_mode = False
                self.deiconify()


if __name__ == "__main__":

    # Instantiate the RL agent (Transformer for White).
    # rl_agent = rlAgent.SimpleAgent(LinearNetwork(), optim.Adam)
    rl_agent = rlAgent.SimpleAgent(SimpleTransformer(), optim.Adam)
    # checkpoint_file = "policy_checkpoint.pth"
    # if os.path.exists(checkpoint_file):
    #     rl_agent.load_checkpoint(checkpoint_file)
    app = ChessApp()
    app.mainloop()
