import copy
import threading
import time
import tkinter as tk

import chess
import torch.optim as optim

import rlAgent
from policyNetwork import LinearNetwork, SimpleTransformer
from utils.classicalAgent import ClassicalAgent
from utils.stockfish import StockfishAgent
from utils.visualBoard import draw_board
import wandb
import torch


#############################################
# GUI Application with Continual RL & Long-Term Rewards
#############################################


class ChessApp(tk.Tk):
    def __init__(self, opponent=None):
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
        self.transformer_name = "Transformer (White)"
        self.classical_name = "Classical (Black)"
        threading.Thread(target=self.game_loop, daemon=True).start()
        if not self.fast_mode:
            self.deiconify()
        self.opponent_agent = opponent_agent

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
                    old_board = copy.deepcopy(self.board)
                    move = rl_agent.choose_move(self.board)
                    self.last_move = move
                    self.board.push(move)
                    new_board = self.board
                    rl_agent.accumulate_immediate_loss(old_board, new_board)

                else:  # Classical Agent as Black.
                    move = self.opponent_agent.get_move(self.board)

                    self.last_move = move
                    self.board.push(move)

                if not self.fast_mode:
                    self.after(0, draw_board(board=self.board, canvas=self.canvas, square_size=self.square_size,
                                             last_move=self.last_move))
                    # Determine status text based on who moved.
                    status_text = (f"{self.transformer_name} moved: {move.uci()}"
                                   if self.board.turn else f"{self.classical_name} moved: {move.uci()}")
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

            run.log({"score": self.model_statistic[-1], "game_count": self.game_count, })

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
        amount_won = len([x for x in self.model_statistic if x == 1])
        amount_draw = len([x for x in self.model_statistic if x == 0])
        amount_lost = len([x for x in self.model_statistic if x == -1])
        run.log({"amount_won": amount_won, "amount_draw": amount_draw, "amount_lost": amount_lost,
                 "games_moves": self.games_moves})


if __name__ == "__main__":
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="lab_course",
        # Set the wandb project where this run will be logged.
        project="ChessLLM",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 1e-3,
            "architecture": "Transformer",
            "dataset": "NONE",
            "epochs": 100,
            "Agent": "piecewise",
        },
    )
    import torch

    # Enable cuDNN autotuner to find the best algorithm for your hardware
    torch.backends.cudnn.benchmark = True
    # Select GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    # Instantiate the Transformer network and move it to GPU
    network = SimpleTransformer().to(device)
    # Instantiate the RL agent (Transformer for White).
    rl_agent = rlAgent.SimpleAgent(LinearNetwork(), optim.AdamW)
    rl_agent = rlAgent.SimpleAgent(network, optim.AdamW)
    # rl_agent = rlAgent.PiecewiseAgent(network, optim.AdamW)

    opponent_agent = ClassicalAgent(depth=3)  # Classical Agent for Black.
    stockfish_path = "stockfish_path"  # Path to your Stockfish binary.
    opponent_agent = StockfishAgent(stockfish_path=stockfish_path)  # Classical Agent for Black.

    # checkpoint_file = "policy_checkpoint.pth"
    # if os.path.exists(checkpoint_file):
    #     rl_agent.load_checkpoint(checkpoint_file)

    app = ChessApp(opponent=opponent_agent)
    app.mainloop()
    run.finish()
