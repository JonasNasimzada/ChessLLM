import chess
from chess.engine import SimpleEngine, Limit


class StockfishAgent:
    def __init__(self, stockfish_path, config={"Skill Level": 20}):
        self.engine = SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure(config)

    def get_move(self, board: chess.Board, time_limit: float = 1.0):
        """
        Get the best move from Stockfish for the given board state.
        :param board: Current chess board state.
        :param time_limit: Time limit for Stockfish to think (in seconds).
        :return: Best move as a chess.Move object.
        """
        result = self.engine.play(board, Limit(time=time_limit))
        print(result)
        return result.move
