"""
This script sets up and runs a chess-playing application where a reinforcement learning (RL) agent competes against a classical chess engine.
The RL agent uses a specified model type (linear, plain transformer, or pretrained transformer) for decision-making.
The results of the games are logged to Weights & Biases (WandB) for analysis.
"""

import argparse
import sys
import threading

import chess
import torch

import wandb
from utils.rlAgent import RLAgent
from utils.classicalAgent import ClassicalAgent


class ChessApp:
    """
    A class to manage and run chess games between an RL agent and a classical chess engine.

    Attributes:
        max_games (int): The maximum number of games to play.
        board (chess.Board): The current chess board state.
        last_move (chess.Move): The last move made in the game.
        opponent (ClassicalAgent): The classical chess engine opponent.
        game_count (int): The number of games played so far.
    """

    def __init__(self, max_games=100):
        """
        Initializes the ChessApp with the specified number of games.

        Args:
            max_games (int): The maximum number of games to play. Default is 100.
        """
        super().__init__()
        self.board = chess.Board()
        self.max_games = max_games
        self.last_move = None
        self.opponent = ClassicalAgent(depth=3)
        self.game_count = 0
        threading.Thread(target=self.game_loop, daemon=True).start()

    def game_loop(self):
        """
        The main game loop that runs the chess games between the RL agent and the classical engine.
        Logs the results of each game to WandB.
        """
        total_win = 0
        total_draw = 0
        total_loss = 0
        while self.game_count < self.max_games:
            self.board.reset()
            self.last_move = None
            self.game_count += 1
            print(f"Starting game {self.game_count}")

            while not self.board.is_game_over():
                if self.board.turn:  # RL (White)
                    move = rl_agent.choose_move(self.board)
                    self.board.push(move)

                    wandb.log({
                        "move_number": len(rl_agent.episode_log_probs),
                        "episode": self.game_count
                    })
                else:  # Classical (Black)
                    move = self.opponent.get_move(self.board)
                    self.board.push(move)

            # End of game.
            result = self.board.result()
            if result == "1-0":
                total_win += 1
            elif result == "0-1":
                total_loss += 1
            else:
                total_draw += 1

            wandb.log({
                "episode": self.game_count,
                "result": result
            })

            print(f"Game {self.game_count} over: {result}")
        wandb.log({
            "total_win": total_win,
            "total_loss": total_loss,
            "total_draw": total_draw
        })


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['linear', 'plain_transformer', 'pretrained_transformer'], default='linear',
                        help='Model type to use for RL agent')
    parser.add_argument('--ckpt', type=str, required=False, help='Path to the model checkpoint file to load')
    parser.add_argument('--max_games', type=int, required=False, default=100, help='Maximum number of games to play')
    args = parser.parse_args()

    # Initialize WandB for logging
    wandb.init(
        project="chess-rl-inference",
        config={
            "learning_rate": 1e-3,
            "reward_weight": 10,
            "minimax_depth": 3,
            "mode": args.model
        }
    )
    sys.stdout = open(f'inference_{args.model}.log', 'w')

    # Load the specified model type
    model = None
    if args.model == "linear":
        from utils.policyNetwork import LinearNetwork
        model = LinearNetwork().to(DEVICE)

    elif args.model == "plain_transformer":
        from utils.policyNetwork import PlainSimpleTransformer
        model = PlainSimpleTransformer().to(DEVICE)
        model.load_state_dict(torch.load(args.ckpt))

    elif args.model == "pretrained_transformer":
        from utils.policyNetwork import LinearLayer, SimpleTransformer
        model = SimpleTransformer().to(DEVICE)
        model = LinearLayer(model).to(DEVICE)
        model.load_state_dict(torch.load(args.ckpt))

    # Initialize the RL agent
    rl_agent = RLAgent(lr=wandb.config.learning_rate, rl_model=model)

    # Track gradients and parameters in WandB
    wandb.watch(rl_agent.policy_net, log="all")

    # Start the chess application
    app = ChessApp(args.max_games)
    app.game_loop()
    wandb.finish()