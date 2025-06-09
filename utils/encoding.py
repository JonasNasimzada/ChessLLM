import chess
import torch

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

def encode_board(board):
    """
    Encode the board as a flattened one-hot tensor.
    Each of 64 squares is represented as a one-hot vector of length 13.
    (Index 0 indicates an empty square.)
    Returns a torch tensor of shape (832,).
    """
    encoding = torch.zeros(64, 13)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.symbol()]
            encoding[square, idx] = 1.0
        else:
            encoding[square, 0] = 1.0
    return encoding.flatten()


def encode_move(move):
    """
    Encode a move as the concatenation of two one-hot vectors for the source and target squares.
    Each one-hot is length 64, so the result is a 128-dimension tensor.
    """
    move_encoding = torch.zeros(128)
    move_encoding[move.from_square] = 1.0
    move_encoding[64 + move.to_square] = 1.0
    return move_encoding


def evaluate_material(board):
    """A simple material evaluation function."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color else -value
    return score
