#!/usr/bin/env python3
"""
Reads a PGN file containing one or more games and writes a CSV where each row
is one half-move (“ply”) with its corresponding FEN string.
"""
import os

import chess
import chess.pgn
import csv


def extract_all_fens_from_pgn(pgn_path: str, csv_path: str) -> None:
    """
    Reads every game in `pgn_path`. For each half-move, writes a row
    [game_index, ply_index, move (UCI), fen_after_move] into `csv_path`.
    """
    with open(pgn_path, encoding="utf-8") as pgn_file, \
            open(csv_path, "w", newline="", encoding="utf-8") as out_csv:

        writer = csv.writer(out_csv)
        # Header: Game # (1-based), Ply # (1-based), Move (UCI), FEN
        writer.writerow(["game_index", "ply_index", "move", "fen"])

        game_index = 0
        i = 0
        while i < 10000:  # Arbitrary large number to prevent infinite loop
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_index += 1
            board = game.board()

            ply_index = 0
            for move in game.mainline_moves():
                ply_index += 1
                fen = board.fen()

                # Get the UCI for this move before pushing
                uci = board.uci(move)

                # Push the move onto the board
                board.push(move)

                # Write one line per half-move
                writer.writerow([game_index, ply_index, uci, fen])
                i += 1


    print(f"Done! Wrote every ply’s UCI and FEN to: {csv_path}")


if __name__ == "__main__":
    # INPUT_PGN_PATH = "./data"
    #
    # for dirpath, dirname, files in os.walk(os.path.dirname(INPUT_PGN_PATH)):
    #     for file in files:
    #         if file.endswith(".pgn"):
    #             extract_all_fens_from_pgn(os.path.join(dirpath, file),
    #                                       os.path.join(dirpath, file.replace(".pgn", ".csv")))

    INPUT_PGN_FILE = "../data_chess/LumbrasGigaBase_OTB_2025.pgn"
    extract_all_fens_from_pgn(INPUT_PGN_FILE, "grpo_data/grpo_Data.csv")
