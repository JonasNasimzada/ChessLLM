import torch
import wandb

from utils import encoding
from utils.encoding import encode_board


#############################################
# RL Agent Definition (Combined Loss per Episode)
#############################################

class RLAgent:
    def __init__(self, policy, optimizer, lr=1e-3):
        self.policy_net = policy
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr)
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
        legal_moves = list(board.legal_moves)
        scores = self.policy_net.calculate_scores(state_vec, legal_moves)
        scores_tensor = torch.stack(scores).view(-1)  # ensure one-dimensional tensor, shape: (n_moves,)
        probs = torch.softmax(scores_tensor, dim=0)
        m = torch.distributions.Categorical(probs)
        index = m.sample()
        log_prob = m.log_prob(index)
        self.episode_log_probs.append(log_prob)
        chosen_move = legal_moves[index.item()]
        return chosen_move

    def finalize_episode(self, final_reward, weight=10):
        """
        At the end of the episode, compute the bonus loss using all stored log probabilities and the final reward,
        add it to cumulative_loss, and then perform one backward update.
        """
        if self.episode_log_probs:
            ep_loss = 0
            for log_prob in self.episode_log_probs:
                ep_loss += -log_prob * final_reward * weight
            self.cumulative_loss += ep_loss
        self.optimizer.zero_grad()
        self.cumulative_loss.backward()
        self.optimizer.step()
        # Reset for the next episode.
        self.episode_log_probs = []
        self.cumulative_loss = 0.0
        wandb.log({'cumulative_loss': self.cumulative_loss})

    def accumulate_immediate_loss(self, old_board, new_board):
        raise NotImplementedError("current RLAgent class did not implement accumulate_immediate_loss")

    def save_checkpoint(self, filename="policy_checkpoint.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print("Checkpoint saved.")

    def load_checkpoint(self, filename="policy_checkpoint.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(filename))
            print("Checkpoint loaded.")
        except Exception as e:
            print("Could not load checkpoint:", e)


class SimpleAgent(RLAgent):
    def __init__(self, policy, optimizer):
        super().__init__(policy, optimizer)

    def accumulate_immediate_loss(self, old_board, new_board):
        """
        Accumulate the immediate loss for the most recent move.
        immediate_loss = - log_prob * immediate_reward.
        Instead of updating immediately, add this to cumulative_loss.
        """
        old_eval = encoding.simple_evaluate_material(old_board)
        new_eval = encoding.simple_evaluate_material(new_board)
        immediate_reward = new_eval - old_eval
        if self.episode_log_probs:
            log_prob = self.episode_log_probs[-1]
            self.cumulative_loss += -log_prob * immediate_reward


class PiecewiseAgent(RLAgent):
    def __init__(self, policy, optimizer):
        super().__init__(policy, optimizer)
        self.immediate_reward = 0.0
        self.gamma = 0.99

    def accumulate_immediate_loss(self, old_board, new_board):
        lost, captured = encoding.evaluate_board_difference_score(old_board, new_board)
        lost_count = -abs(sum(encoding.piece_reward[piece] for piece in lost))
        captured_count = sum(encoding.piece_reward[piece] for piece in captured)
        self.immediate_reward = torch.tensor((lost_count + captured_count) * self.gamma, requires_grad=True)
        wandb.log({'immediate_reward': self.immediate_reward})
        self.optimizer.zero_grad()
        self.immediate_reward.backward()
        self.optimizer.step()
