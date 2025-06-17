# encode_utils.py
import chess, torch


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
# State and Action Encoding Functions
#############################################


def encode_board(fen_or_board):
    """
    Encode the board as a flattened one-hot tensor.
    Each of 64 squares is represented as a one-hot vector of length 13.
    (Index 0 indicates an empty square.)
    Returns a torch tensor of shape (832,).
    """

    if isinstance(fen_or_board, str):
        board = chess.Board(fen_or_board)
    else:
        board = fen_or_board

    encoding = torch.zeros(64, 13)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.symbol()]
            encoding[square, idx] = 1.0
        else:
            encoding[square, 0] = 1.0
    return encoding.flatten()


def encode_move(move_or_obj):
    """
    Encode a move as the concatenation of two one-hot vectors for the source and target squares.
    Each one-hot is length 64, so the result is a 128-dimension tensor.
    """

    if isinstance(move_or_obj, str):
        move = chess.Move.from_uci(move_or_obj)
    else:
        move = move_or_obj

    move_encoding = torch.zeros(128)
    move_encoding[move.from_square] = 1.0
    move_encoding[64 + move.to_square] = 1.0
    return move_encoding


# encode_utils.py 末尾（或合适位置）追加
import chess
from typing import List

def decode_move(idx: int, legal_moves: List[chess.Move]) -> chess.Move:
    """
    Map an integer index back to the concrete chess.Move object.

    We treat `idx` as the index into the *current* list of legal moves.
    This stays consistent with how RLAgent builds the (board+move) tensor
    for each move in `legal_moves`.

    Parameters
    ----------
    idx : int
        Index in the legal move list.
    legal_moves : list[chess.Move]
        The list produced by `board.legal_moves`.

    Returns
    -------
    chess.Move
        The concrete move corresponding to that index.

    Raises
    ------
    IndexError
        If idx is out of range.
    """
    return list(legal_moves)[idx]
