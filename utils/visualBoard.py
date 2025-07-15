import chess

from utils.encoding import piece_unicode


def draw_board(board: chess.Board, canvas, square_size, last_move=None):
    """
    Draws a chess board on the given canvas, including pieces and highlighting the last move.

    Args:
        board (chess.Board): The current state of the chess board.
        canvas: The canvas object where the board will be drawn.
        square_size (int): The size of each square on the board in pixels.
        last_move (chess.Move, optional): The last move made on the board. If provided,
                                          the move will be highlighted. Defaults to None.
    """
    # Clear the canvas
    canvas.delete("all")

    # Iterate through each square on the board
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            x1 = file * square_size
            y1 = (7 - rank) * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size

            # Determine the square color (light or dark)
            color = "bisque" if (file + rank) % 2 == 0 else "sienna"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color)

            # Draw the piece if one exists on the square
            piece = board.piece_at(square)
            if piece:
                symbol = piece_unicode[piece.symbol()]
                canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                   text=symbol, font=("Arial", 36))

    # Highlight the last move if provided
    if last_move:
        for sq in [last_move.from_square, last_move.to_square]:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            x1 = file * square_size
            y1 = (7 - rank) * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=3)