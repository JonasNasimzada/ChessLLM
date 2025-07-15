import random

from utils.encoding import simple_evaluate_material


#############################################
# Classical Agent (Minimax with Alpha-Beta)
#############################################
class ClassicalAgent:
    """
    A classical chess agent that uses the Minimax algorithm with Alpha-Beta pruning
    to determine the best move.

    Attributes:
        depth (int): The depth of the Minimax search tree.
    """

    def __init__(self, depth=3):
        """
        Initializes the ClassicalAgent with a specified search depth.

        Args:
            depth (int): The depth of the Minimax search tree. Default is 3.
        """
        self.depth = depth

    def get_move(self, board):
        """
        Determines the best move for the current board state using the Minimax algorithm.

        Args:
            board (chess.Board): The current chess board state.

        Returns:
            chess.Move: The chosen move for the agent.
        """
        legal_moves = list(board.legal_moves)
        best_moves = []
        best_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            current_eval = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), False)
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

    def minimax(self, board, depth, alpha, beta, maximizing):
        """
        Implements the Minimax algorithm with Alpha-Beta pruning to evaluate board states.

        Args:
            board (chess.Board): The current chess board state.
            depth (int): The remaining depth of the search tree.
            alpha (float): The best value that the maximizer currently can guarantee.
            beta (float): The best value that the minimizer currently can guarantee.
            maximizing (bool): True if the current player is maximizing, False otherwise.

        Returns:
            float: The evaluation score of the board state.
        """
        if depth == 0 or board.is_game_over():
            return simple_evaluate_material(board)
        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta, False)
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
                eval_val = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval
