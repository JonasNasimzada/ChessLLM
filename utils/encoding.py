import re

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

piece_reward = {
    chess.PAWN: 1,
    chess.KNIGHT: 5,
    chess.BISHOP: 5,
    chess.ROOK: 5,
    chess.QUEEN: 10,
    chess.KING: 100
}


#############################################
# State and Action Encoding Functions
#############################################

def encode_board(board):
    """
    Encode the board as a flattened one-hot tensor.
    Each of the 64 squares is represented as a one-hot vector of length 13.
    (Index 0 indicates an empty square.)

    Args:
        board (chess.Board): The chess board to encode.

    Returns:
        torch.Tensor: A tensor of shape (832,) representing the board.
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
    Each one-hot vector is of length 64, resulting in a 128-dimensional tensor.

    Args:
        move (chess.Move): The chess move to encode.

    Returns:
        torch.Tensor: A tensor of shape (128,) representing the move.
    """
    move_encoding = torch.zeros(128)
    move_encoding[move.from_square] = 1.0
    move_encoding[64 + move.to_square] = 1.0
    return move_encoding


def simple_evaluate_material(board):
    """
    Evaluate the material balance on the board.

    Args:
        board (chess.Board): The chess board to evaluate.

    Returns:
        int: The material score of the board.
    """
    score = 0
    score_new = 0
    piece_list = count_pieces(board)
    for color, piece_type in piece_list:
        value = piece_values[piece_type]
        score_new += (value if color else -abs(value)) * piece_list[color, piece_type]
    return score


def count_pieces(board):
    """
    Count the pieces on the board.

    Args:
        board (chess.Board): The chess board to analyze.

    Returns:
        dict: A dictionary with counts of pieces by color and type.
    """
    cnt = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            key = (piece.color, piece.piece_type)
            cnt[key] = cnt.get(key, 0) + 1
    return cnt


def evaluate_board_difference_score(old_board, new_board):
    """
    Calculate the difference in material between two boards.

    Args:
        old_board (chess.Board): The board before the move.
        new_board (chess.Board): The board after the move.

    Returns:
        tuple: Two lists, one for lost pieces and one for captured pieces.
    """
    old_counts = count_pieces(old_board)
    new_counts = count_pieces(new_board)

    lost = []
    captured = []

    for piece_type in chess.PIECE_TYPES:
        old_black = old_counts.get((chess.BLACK, piece_type), 0)
        new_black = new_counts.get((chess.BLACK, piece_type), 0)
        if new_black < old_black:
            captured.extend([piece_type for _ in range(old_black - new_black)])

        old_white = old_counts.get((chess.WHITE, piece_type), 0)
        new_white = new_counts.get((chess.WHITE, piece_type), 0)
        if new_white < old_white:
            lost.extend([piece_type for _ in range(old_white - new_white)])
    return lost, captured


FEN_REGEX = (
    r'^\s*'
    r'('
    r'(?:[rnbqkpRNBQKP1-8]+\/){7}[rnbqkpRNBQKP1-8]+'  # board part
    r'\s[bw]'  # side
    r'\s(?:[KQkq]{1,4}|-)'  # castling
    r'\s(?:-|[a-h][1-8])'  # en passant
    r'\s\d+\s\d+'  # move counters
    r')'
    r'\s*$'
)
UCI_REGEX = r'\b([a-h][1-8][a-h][1-8][nbrq]?)\b'


def isolate_fen_notation(prompt):
    """
    Extract the FEN notation from a given prompt.

    Args:
        prompt (str): The input prompt containing FEN notation.

    Returns:
        str or None: The extracted FEN string, or None if not found.
    """
    pattern = re.compile(FEN_REGEX, re.MULTILINE)
    search = pattern.findall(prompt)
    if search:
        fen = search[-1]
        return fen
    else:
        return None


def isolate_move_notation(response):
    """
    Extract the UCI move notation from a given response.

    Args:
        response (str): The input response containing UCI move notation.

    Returns:
        str or None: The extracted UCI move string, or None if not found.
    """
    search = re.search(UCI_REGEX, response)
    if search:
        uci = search.group(1)
        return uci
    else:
        return None
