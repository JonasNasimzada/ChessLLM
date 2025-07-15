import torch
import wandb

from utils import encoding
from utils.encoding import encode_board


#############################################
# RL Agent Definition (Combined Loss per Episode)
#############################################

class RLAgent:
    """
    A reinforcement learning agent that uses a policy network to choose moves
    and optimizes the policy using cumulative loss over an episode.

    Args:
        policy (nn.Module): The policy network used to evaluate moves.
        optimizer (torch.optim.Optimizer): The optimizer for training the policy network.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(self, policy, optimizer, lr=1e-3):
        self.policy_net = policy
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr)
        self.episode_log_probs = []  # Stores log probabilities of moves during the episode.
        self.cumulative_loss = 0.0  # Accumulates loss over the episode.

    def choose_move(self, board):
        """
        Selects a move from the legal moves of the given board using the policy network.

        Args:
            board (chess.Board): The current chess board.

        Returns:
            chess.Move: The chosen move.
        """
        state_vec = encode_board(board)  # Encodes the board state as a tensor.
        legal_moves = list(board.legal_moves)  # Retrieves all legal moves.
        scores = self.policy_net.calculate_scores(state_vec, legal_moves)  # Scores for each move.
        probs = torch.softmax(scores, dim=0)  # Converts scores to probabilities.
        m = torch.distributions.Categorical(probs)  # Creates a categorical distribution.
        index = m.sample()  # Samples a move index.
        log_prob = m.log_prob(index)  # Logs the probability of the chosen move.
        self.episode_log_probs.append(log_prob)  # Stores the log probability.
        chosen_move = legal_moves[index.item()]  # Retrieves the chosen move.
        return chosen_move

    def finalize_episode(self, final_reward, weight=10):
        """
        Finalizes the episode by computing the bonus loss, updating the cumulative loss,
        and performing a backward update on the policy network.

        Args:
            final_reward (float): The reward at the end of the episode.
            weight (float): A scaling factor for the final reward. Default is 10.
        """
        if self.episode_log_probs:
            ep_loss = 0
            for log_prob in self.episode_log_probs:
                ep_loss += -log_prob * final_reward * weight  # Computes the episode loss.
            self.cumulative_loss += ep_loss  # Adds to the cumulative loss.
        self.optimizer.zero_grad()  # Resets gradients.
        self.cumulative_loss.backward()  # Computes gradients.
        self.optimizer.step()  # Updates the policy network.
        wandb.log({'cumulative_loss': self.cumulative_loss})  # Logs the cumulative loss.
        self.episode_log_probs = []  # Resets log probabilities for the next episode.
        self.cumulative_loss = 0.0  # Resets cumulative loss.

    def accumulate_immediate_loss(self, old_board, new_board):
        """
        Abstract method to accumulate immediate loss for a move.
        Must be implemented by subclasses.

        Args:
            old_board (chess.Board): The board state before the move.
            new_board (chess.Board): The board state after the move.
        """
        raise NotImplementedError("current RLAgent class did not implement accumulate_immediate_loss")

    def save_checkpoint(self, filename="policy_checkpoint.pth"):
        """
        Saves the policy network's state to a file.

        Args:
            filename (str): The file path to save the checkpoint. Default is "policy_checkpoint.pth".
        """
        torch.save(self.policy_net.state_dict(), filename)
        print("Checkpoint saved.")

    def load_checkpoint(self, filename="policy_checkpoint.pth"):
        """
        Loads the policy network's state from a file.

        Args:
            filename (str): The file path to load the checkpoint. Default is "policy_checkpoint.pth".
        """
        try:
            self.policy_net.load_state_dict(torch.load(filename))
            print("Checkpoint loaded.")
        except Exception as e:
            print("Could not load checkpoint:", e)


class SimpleAgent(RLAgent):
    """
    A simple reinforcement learning agent that accumulates immediate loss
    based on material evaluation differences between board states.
    """

    def __init__(self, policy, optimizer):
        super().__init__(policy, optimizer)

    def accumulate_immediate_loss(self, old_board, new_board):
        """
        Accumulates immediate loss for the most recent move based on material evaluation.

        Args:
            old_board (chess.Board): The board state before the move.
            new_board (chess.Board): The board state after the move.
        """
        old_eval = encoding.simple_evaluate_material(old_board)  # Material evaluation of the old board.
        new_eval = encoding.simple_evaluate_material(new_board)  # Material evaluation of the new board.
        immediate_reward = new_eval - old_eval  # Computes the immediate reward.
        if self.episode_log_probs:
            log_prob = self.episode_log_probs[-1]  # Retrieves the log probability of the last move.
            self.cumulative_loss += -log_prob * immediate_reward  # Updates the cumulative loss.


class PiecewiseAgent(RLAgent):
    """
    A reinforcement learning agent that accumulates immediate loss based on
    piecewise rewards for captured and lost pieces.
    """

    def __init__(self, policy, optimizer):
        super().__init__(policy, optimizer)
        self.immediate_reward = 0.0  # Stores the immediate reward for the current move.
        self.gamma = 0.99  # Discount factor for rewards.

    def accumulate_immediate_loss(self, old_board, new_board):
        """
        Accumulates immediate loss for the most recent move based on piecewise rewards.

        Args:
            old_board (chess.Board): The board state before the move.
            new_board (chess.Board): The board state after the move.
        """
        lost, captured = encoding.evaluate_board_difference_score(old_board, new_board)  # Evaluates board differences.
        lost_count = -abs(sum(encoding.piece_reward[piece] for piece in lost))  # Computes lost piece rewards.
        captured_count = sum(encoding.piece_reward[piece] for piece in captured)  # Computes captured piece rewards.

        self.immediate_reward = (lost_count + captured_count) * self.gamma  # Computes the immediate reward.
        wandb.log({'immediate_reward': self.immediate_reward})  # Logs the immediate reward.
        self.immediate_reward = torch.tensor((lost_count + captured_count) * self.gamma, requires_grad=True)

        self.optimizer.zero_grad()  # Resets gradients.
        self.immediate_reward.backward()  # Computes gradients for the immediate reward.
        self.optimizer.step()  # Updates the policy network.
