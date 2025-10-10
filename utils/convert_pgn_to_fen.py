#!/usr/bin/env python3
"""
Reads a PGN file containing one or more games and writes a CSV where each row
is one half-move (“ply”) with its corresponding FEN string.
"""
import argparse
import csv

import chess
import chess.pgn


def count_pieces(board):
    piece_count = 0
    for pieces in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece_count += len(board.pieces(pieces, color))
    return piece_count


def extract_all_fens_from_pgn(pgn_path: str, amount_datapoints: int = 1_000_000):
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
    endgame = []
    tactic = []
    midgame = []
    opening_game = []

    fullgames = []

    game_parts = [endgame, tactic, midgame, opening_game]
    filenames = [f"{args.output}_endgame.csv", f"{args.output}_tactic.csv", f"{args.output}_midgame.csv",
                 f"{args.output}_opening_game.csv"]
    fullgames_filename = f"{args.output}.csv"

    with open(pgn_path, encoding="utf-8") as pgn_file:
        game_index = 0
        i = 0
        while i < amount_datapoints or (args.split and len(endgame) < args.size_endgame):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_index += 1
            board = game.board()

            ply_index = 0
            for move in game.mainline_moves():

                if i < args.start:
                    board.push(move)
                    continue
                i += 1

                ply_index += 1
                fen = board.fen()

                # Get the UCI for this move before pushing
                uci = board.uci(move)
                # Push the move onto the board
                board.push(move)
                if args.split:
                    if count_pieces(board) <= 5:
                        endgame.append((game_index, ply_index, uci, fen))
                    elif count_pieces(board) <= 10:
                        tactic.append((game_index, ply_index, uci, fen))
                    elif count_pieces(board) <= 20:
                        midgame.append((game_index, ply_index, uci, fen))
                    else:
                        opening_game.append((game_index, ply_index, uci, fen))

                fullgames.append((game_index, ply_index, uci, fen))

                if i % 100000 == 0:
                    print(f"Extracted {i} FENs so far...")

    if args.split:
        for i in range(len(game_parts)):
            with open(filenames[i], mode='a', newline="", encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['game_index', 'ply_index', 'move', 'fen'])
                writer.writerows(game_parts[i])
            print(f"Done! Wrote every ply’s UCI and FEN to: {filenames[i]}, Datapoints: {len(game_parts[i])}")
    with open(fullgames_filename, mode='w', newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['game_index', 'ply_index', 'move', 'fen'])
        writer.writerows(fullgames)
        print(f"Done! Wrote every ply’s UCI and FEN to: {fullgames_filename}, Datapoints: {len(fullgames)}")



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
        default="LumbrasGigaBase_OTB_2025",
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
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        required=False,
        help="Starting index for extraction"
    )
    parser.add_argument(
        "--split",
        type=bool,
        default=True,
        required=False,
        help="Split into csv depending on game phase (endgame, midgame, opening, tactic, fullgames)."
    )
    parser.add_argument(
        "--size_endgame",
        type=int,
        default=1_000_000,
        required=False,
        help="Maximum number of FENs to extract from the PGN file to endgame."
    )

    args = parser.parse_args()
    extract_all_fens_from_pgn(args.data, args.size)
