#!/usr/bin/env python3
"""
Reads a PGN file containing one or more games and writes a CSV where each row
is one half-move (“ply”) with its corresponding FEN string.
"""

import chess
import chess.pgn
import csv

INPUT_PGN_PATH = "../trainings_data/LumbrasGigaBase 2024.pgn"
OUTPUT_CSV_PATH = "../trainings_data/LumbrasGigaBase 2024.csv"


def extract_all_fens_from_pgn(pgn_path: str, csv_path: str) -> None:
    """
    Reads every game in `pgn_path`. For each half-move, writes a row
    [game_index, ply_index, move (SAN), fen_after_move] into `csv_path`.
    """
    with open(pgn_path, encoding="utf-8") as pgn_file, \
            open(csv_path, "w", newline="", encoding="utf-8") as out_csv:

        writer = csv.writer(out_csv)
        # Header: Game # (1-based), Ply # (1-based), Move (SAN), FEN
        writer.writerow(["game_index", "ply_index", "move", "fen"])

        game_index = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_index += 1
            board = game.board()

            ply_index = 0
            for move in game.mainline_moves():
                ply_index += 1

                # Get the SAN for this move before pushing
                san = board.san(move)

                # Push the move onto the board
                board.push(move)

                # Get FEN after this move
                fen = board.fen()

                # Write one line per half-move
                writer.writerow([game_index, ply_index, san, fen])

    print(f"Done! Wrote every ply’s SAN and FEN to: {csv_path}")


if __name__ == "__main__":
    extract_all_fens_from_pgn(INPUT_PGN_PATH, OUTPUT_CSV_PATH)
