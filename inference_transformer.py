import argparse
import sys
import threading

import chess
import torch

import wandb
from utils.rlAgent import RLAgent
from utils.classicalAgent import ClassicalAgent


class ChessApp:
    def __init__(self, max_games=100):
        super().__init__()
        self.board = chess.Board()
        self.max_games = max_games
        self.last_move = None
        self.opponent = ClassicalAgent(depth=3)
        self.game_count = 0
        threading.Thread(target=self.game_loop, daemon=True).start()

    def game_loop(self):
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['linear', 'plain_transformer', 'pretrained_transformer'], default='linear',
                        help='Model type to use for RL agent')
    parser.add_argument('--ckpt', type=str, required=False)
    parser.add_argument('--max_games', type=int, required=False, default=100)
    args = parser.parse_args()

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

    rl_agent = RLAgent(lr=wandb.config.learning_rate, rl_model=model)

    # Track gradients & parameters
    wandb.watch(rl_agent.policy_net, log="all")

    app = ChessApp(args.max_games)
    app.game_loop()
    wandb.finish()

