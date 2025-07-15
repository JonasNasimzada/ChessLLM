#!/usr/bin/env python3
"""
Reads a PGN file containing one or more games and writes a CSV where each row
is one half-move (“ply”) with its corresponding FEN string.
"""
import argparse
import csv

import chess
import chess.pgn


def extract_all_fens_from_pgn(pgn_path: str, csv_path: str, amount_datapoints: int = 1_000_000) -> None:
    """
    Extracts FEN strings and corresponding move data from a PGN file and writes them to a CSV file.

    Args:
        pgn_path (str): Path to the input PGN file containing chess games.
        csv_path (str): Path to the output CSV file where FENs and moves will be saved.
        amount_datapoints (int): Maximum number of FENs to extract. Default is 1,000,000.

    Writes:
        A CSV file with the following columns:
        - game_index: The index of the game (1-based).
        - ply_index: The index of the half-move (1-based).
        - move: The move in UCI format.
        - fen: The FEN string after the move.
    """
    with open(pgn_path, encoding="utf-8") as pgn_file, \
            open(csv_path, "w", newline="", encoding="utf-8") as out_csv:

        writer = csv.writer(out_csv)
        # Header: Game # (1-based), Ply # (1-based), Move (UCI), FEN
        writer.writerow(["game_index", "ply_index", "move", "fen"])

        game_index = 0
        i = 0
        while i < amount_datapoints:
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
    """
    Entry point for the script. Parses command-line arguments and extracts FENs from the specified PGN file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="LumbrasGigaBase_OTB_2025.pgn",
        required=False,
        help="Path to the input PGN file containing chess games."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="LumbrasGigaBase_OTB_2025.csv",
        required=False,
        help="Path to the output CSV file where FENs will be saved."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1_000_000,
        required=False,
        help="Maximum number of FENs to extract from the PGN file."
    )

    args = parser.parse_args()
    extract_all_fens_from_pgn(args.data, args.output, args.size)
