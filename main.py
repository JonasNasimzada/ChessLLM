import threading
import time
import tkinter as tk

import chess
import torch.optim as optim

from utils.classicalAgent import classical_agent_move
from policyNetwork import LinearNetwork, SimpleTransformer
import rlAgent
from utils.encoding import evaluate_material
from utils.visualBoard import draw_board

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

    def game_loop(self):
        checkpoint_file = "policy_checkpoint.pth"
        # while True:
        while self.game_count < 101:
            self.board.reset()
            self.last_move = None
            self.game_count += 1
            print(f"Starting game {self.game_count}")
            self.status_label.config(text=f"Game {self.game_count} start")
            # Clear accumulated losses for a new episode.
            rl_agent.episode_log_probs = []
            rl_agent.cumulative_loss = 0.0
            amount_moves = 0

            # Play one complete game.
            while not self.board.is_game_over():
                amount_moves += 1
                if self.board.turn:  # Transformer (RL Agent) as White.
                    transformer_name = "Transformer (White)"
                    move = rl_agent.choose_move(self.board)
                    self.last_move = move
                    self.board.push(move)

                else:  # Classical Agent as Black.
                    move = classical_agent_move(self.board, depth=3)
                    classical_name = "Classical (Black)"
                    self.last_move = move
                    self.board.push(move)

                if not self.fast_mode:
                    self.after(0, draw_board(board=self.board, canvas=self.canvas, square_size=self.square_size,
                                             last_move=self.last_move))
                    # Determine status text based on who moved.
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
                classical_result = "drew"
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
            # if self.game_count == 10 and self.fast_mode:
            #     print("Completed 100 games in fast mode; switching to UI mode with delays.")
            #     self.fast_mode = False
            #     self.deiconify()
        print(f"The Agent won:{[x for x in self.model_statistic if x == 1]} "
              f"draw: {[x for x in self.model_statistic if x == 0]} "
              f"lost: {[x for x in self.model_statistic if x == -1]}")


if __name__ == "__main__":
    # Instantiate the RL agent (Transformer for White).
    rl_agent = rlAgent.SimpleAgent(LinearNetwork(), optim.AdamW)
    rl_agent = rlAgent.SimpleAgent(SimpleTransformer(), optim.AdamW)
    # checkpoint_file = "policy_checkpoint.pth"
    # if os.path.exists(checkpoint_file):
    #     rl_agent.load_checkpoint(checkpoint_file)
    app = ChessApp()
    app.mainloop()
