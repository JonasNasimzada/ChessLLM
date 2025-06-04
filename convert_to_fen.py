#!/usr/bin/env python3
"""
Script: pgn_to_fen_csv.py

Reads a PGN file containing one or more games and writes a CSV where each row
is one half-move (“ply”) with its corresponding FEN string.

Requires: python-chess
    pip install python-chess
"""

import chess
import chess.pgn
import csv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PGN_PATH = "LumbrasGigaBase 2024.pgn"  # Path to your input PGN file
OUTPUT_CSV_PATH = "LumbrasGigaBase 2024.csv"  # Path for the output CSV


# ───────────────────────────────────────────────────────────────────────────────


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
