import random
import threading
import time
import tkinter as tk

import chess
import torch
import torch.optim as optim

from encode_utils import encode_board, encode_move, piece_unicode, piece_values
from llm_agent   import get_llm_board_eval
from policyNetwork import SimpleTransformer
import rlAgent


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
        self.model_statistic = []
        self.games_moves = []
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
            rl_agent.reset_episode_buffers()

            amount_moves = 0

            # Play one complete game.
            while not self.board.is_game_over():
                amount_moves += 1
                if self.board.turn:  # White to play
                    old_eval = evaluate_material(self.board)

                    # -----------------------------------------------------------------
                    # 1)  Build one (state + move) tensor for every legal move
                    #     -> each tensor has 960 dims = 832 (board) + 128 (move)
                    # -----------------------------------------------------------------
                    legal_moves = list(self.board.legal_moves)
                    state_vec = encode_board(self.board)  # (832,)
                    scores = []
                    for mv in legal_moves:
                        mv_vec = encode_move(mv)  # (128,)
                        inp = torch.cat([state_vec, mv_vec]).unsqueeze(0)  # (1,960)
                        logits = rl_agent.policy_net(inp).squeeze()  # (128,)
                        score = logits.mean()  # scalar value
                        scores.append(score)

                    scores_tensor = torch.stack(scores)  # shape (N,)

                    # -----------------------------------------------------------------
                    # 2)  Pick the Transformer’s top-k moves
                    # -----------------------------------------------------------------
                    k = min(5, len(legal_moves))
                    top_indices = scores_tensor.topk(k).indices.tolist()   # List[int]

                    # -----------------------------------------------------------------
                    # 3)  Ask the LLM to evaluate the positions after each candidate move
                    # -----------------------------------------------------------------
                    llm_scores = []
                    for idx in top_indices:  # idx is now plain int
                        mv = legal_moves[idx]
                        self.board.push(mv)
                        llm_scores.append(get_llm_board_eval(self.board.fen()))
                        self.board.pop()

                    # -----------------------------------------------------------------
                    # 4)  Fuse the scores (simple sum; can be weighted if desired)
                    # -----------------------------------------------------------------
                    final_scores = scores_tensor.clone()
                    for offset, idx in enumerate(top_indices):
                        final_scores[idx] += llm_scores[offset]

                    # -----------------------------------------------------------------
                    # 5)  Select and execute the move with the highest combined score
                    # -----------------------------------------------------------------
                    best_idx = final_scores.argmax().item()
                    move = legal_moves[best_idx]

                    self.board.push(move)
                    self.last_move = move

                    # -----------------------------------------------------------------
                    # 6)  Compute dense reward and pass it to the RL agent
                    # -----------------------------------------------------------------
                    new_eval = evaluate_material(self.board)
                    rl_agent.accumulate_immediate_loss(new_eval - old_eval)


                else:  # Classical Agent as Black.
                    move = classical_agent_move(self.board, depth=3)
                    classical_name = "Classical (Black)"
                    self.last_move = move
                    self.board.push(move)
                if not self.fast_mode:
                    self.after(0, self.draw_board)
                    # Determine status text based on who moved.
                    transformer_name = "Qwen/Qwen1.5-0.5B-Chat"
                    status_text = (f"{transformer_name} moved: {move.uci()}"
                                   if self.board.turn else f"{classical_name} moved: {move.uci()}")
                    self.after(0, lambda text=status_text: self.status_label.config(text=text))
                    time.sleep(0.5)
            # End of game.
            result = self.board.result()  # "1-0", "0-1", or "1/2-1/2"
            self.games_moves.append(amount_moves)

            print(f"Game {self.game_count} over: {result}")
            if result == "1-0":
                final_reward = 10
                transformer_result = "won"
                classical_result = "lost"
                self.model_statistic.append(1)

            elif result == "0-1":
                final_reward = -10
                transformer_result = "lost"
                classical_result = "won"
                self.model_statistic.append(-1)
            else:
                final_reward = -10
                transformer_result = "drew (punished)"
                classical_result = "drew (punished)"
                self.model_statistic.append(0)

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
