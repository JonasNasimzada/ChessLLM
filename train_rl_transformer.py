import argparse
import copy
import threading
import time
import tkinter as tk

import chess
import torch.optim as optim

from utils import rlAgent
import wandb
from utils.classicalAgent import ClassicalAgent
from utils.visualBoard import draw_board


#############################################
# GUI Application with Continual RL & Long-Term Rewards
#############################################


class ChessApp(tk.Tk):
    """
    A GUI application for playing chess between a Reinforcement Learning (RL) Transformer agent
    and a Classical Chess agent. The application supports continual learning and long-term rewards.

    Attributes:
        opponent_agent (ClassicalAgent): The opponent agent (Classical Chess agent).
        checkpoint_file (str): Path to the checkpoint file for saving the RL agent's state.
        fast_mode (bool): If True, the GUI is hidden, and the game runs in the background.
        board (chess.Board): The chess board object.
        square_size (int): Size of each square on the chessboard.
        canvas (tk.Canvas): Canvas for drawing the chessboard.
        status_label (tk.Label): Label for displaying the game status.
        last_move (chess.Move): The last move made in the game.
        game_count (int): Counter for the number of games played.
        model_statistic (list): List to store the results of each game (1 for win, -1 for loss, 0 for draw).
        games_moves (list): List to store the number of moves in each game.
        transformer_name (str): Name of the RL Transformer agent.
        classical_name (str): Name of the Classical Chess agent.
    """

    def __init__(self, opponent=None, checkpoint="policy_checkpoint.pth", fast_mode=False):
        """
        Initializes the ChessApp.

        Args:
            opponent (ClassicalAgent): The opponent agent (Classical Chess agent).
            checkpoint (str): Path to the checkpoint file for saving the RL agent's state.
            fast_mode (bool): If True, the GUI is hidden, and the game runs in the background.
        """
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
        self.fast_mode = fast_mode
        self.game_count = 0
        self.withdraw()  # Hide the window in fast mode.
        self.model_statistic = []
        self.games_moves = []
        self.transformer_name = "Transformer (White)"
        self.classical_name = "Classical (Black)"
        self.checkpoint_file = checkpoint
        threading.Thread(target=self.game_loop, daemon=True).start()
        if not self.fast_mode:
            self.deiconify()
        self.opponent_agent = opponent_agent

    def game_loop(self):
        """
        Main game loop for playing multiple games between the RL Transformer agent and the Classical agent.
        The loop continues until 101 games are played.
        """
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
                    self.after(1, draw_board(board=self.board, canvas=self.canvas, square_size=self.square_size,
                                             last_move=self.last_move))
                    status_text = (f"{self.transformer_name} moved: {move.uci()}"
                                   if self.board.turn else f"{self.classical_name} moved: {move.uci()}")
                    self.after(0, lambda text=status_text: self.status_label.config(text=text))
                    time.sleep(1.0)
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
            rl_agent.save_checkpoint(filename=self.checkpoint_file)
            if not self.fast_mode:
                time.sleep(3)
        amount_won = len([x for x in self.model_statistic if x == 1])
        amount_draw = len([x for x in self.model_statistic if x == 0])
        amount_lost = len([x for x in self.model_statistic if x == -1])
        run.log({"amount_won": amount_won, "amount_draw": amount_draw, "amount_lost": amount_lost,
                 "games_moves": self.games_moves})


if __name__ == "__main__":
    """
    Entry point for the application. Parses command-line arguments, initializes the RL agent,
    opponent agent, and starts the ChessApp GUI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['simple', 'piecewise'], default='piecewise', required=False, )
    parser.add_argument('--model', choices=['linear', 'plain_transformer', 'pretrained_transformer'], default='linear',
                        help='Model type to use for RL agent')
    parser.add_argument('--ckpt', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    args = parser.parse_args()

    run = wandb.init(
        entity="lab_course",
        project="rl_chess_transformer",
        config={
            "learning_rate": 1e-3,
            "architecture": args.model,
            "epochs": args.epochs,
            "Agent": args.model,
        },
    )
    import torch

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = None
    if args.model == "linear":
        from utils.policyNetwork import LinearNetwork

        network = LinearNetwork().to(DEVICE)

    elif args.model == "plain_transformer":
        from utils.policyNetwork import PlainSimpleTransformer

        network = PlainSimpleTransformer().to(DEVICE)

    elif args.model == "pretrained_transformer":
        from utils.policyNetwork import LinearLayer, SimpleTransformer

        network = SimpleTransformer().to(DEVICE)
        network = LinearLayer(network).to(DEVICE)
        network.load_state_dict(torch.load(args.ckpt))

    rl_agent = None
    if args.model == "simple":

        rl_agent = rlAgent.SimpleAgent(network, optim.AdamW)
    elif args.rl_agent == "piecewise":

        rl_agent = rlAgent.PiecewiseAgent(network, optim.AdamW)

    opponent_agent = ClassicalAgent(depth=3)
    checkpoint_file = f"rl_transformer_{args.model}_{args.model}.pth"

    app = ChessApp(opponent=opponent_agent, checkpoint=checkpoint_file)
    app.mainloop()
    app.quit()
    run.finish()
