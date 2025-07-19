"""This script implements a chess-playing engine that uses a combination of reinforcement learning (RL) and classical
chess engines (e.g., Stockfish). It defines functions for making moves, generating moves using a language model,
and playing chess games while logging results to Weights & Biases (WandB)."""
import os
import random
import time
from collections import deque

import chess
import torch
import wandb
from stockfish import Stockfish
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from utils.classicalAgent import ClassicalAgent
from utils.encoding import isolate_move_notation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# System message template for the RL agent
system_message = """You are the world’s strongest chess engine. You will be given the full move-history in FEN 
notation followed by the current position in FEN. Your task is to think through the position step by step—evaluating 
piece placement, pawn structure, king safety, candidate moves and tactical motifs—and then output exactly one best 
move in UCI format.\n\nStep-by-step guide:\n1. Material count and piece activity\n2. Pawn structure and central 
control\n3. King safety for both sides\n4. Candidate moves (e.g. developing, challenging the bishop, castling)\n5. 
Tactical considerations (pins, forks, discovered attacks)\n6. Long-term strategic plans\n\nAfter reasoning, 
output only the best move in UCI format.Respond in the following format: <think> You should reason between these 
tags. </think>\n The resulting UCI move should be between <answer> </answer> tags\n Always use <think> </think> tags 
even if they are not necessary."""

# User message template for prompting the RL agent
user_message = """Move history (in FEN):\n{past_moves}\n\nCurrent position (FEN):\n{current_move}\n\nWhat is the next 
best move in UCI format?"""
user_message_no_context = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""


def engine_make_move(current_board, past_fen_moves, engine="stockfish"):
    """
    Makes a move using the specified chess engine (Stockfish or Minimax).

    Args:
        current_board (chess.Board): The current chess board state.
        past_fen_moves (deque): A deque containing the FEN strings of past moves.
        engine (str): The engine to use for move generation ("stockfish" or "minmax").

    Returns:
        bool: True if the board state was reset due to invalid moves, False otherwise.
    """
    if not current_board.is_valid():
        print("Invalid board state, resetting last two moves.")
        if past_fen_moves:
            for _ in range(2):
                current_board.pop()
                if past_fen_moves:
                    past_fen_moves.pop()
        return True

    move = None
    if engine == "stockfish":
        fen = current_board.fen()
        past_fen_moves.append(fen)
        stockfish_agent.set_fen_position(fen)
        move = chess.Move.from_uci(stockfish_agent.get_best_move_time(100))
    elif engine == "minmax":
        move = ClassicalAgent(depth=3).get_move(board=current_board)

    current_board.push(move)
    return False


def generate_move(prompt):
    """
    Generates a move using the RL model based on the given prompt.

    Args:
        prompt (list): A list of dictionaries containing the system and user messages.

    Returns:
        str: The best move in UCI format.
    """
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    move_str = tokenizer.decode(output[0], skip_special_tokens=True)
    move = isolate_move_notation(move_str)
    return move


def generate_prompt(past_moves, current_board):
    """
    Generates a prompt for the RL agent based on past moves and the current board state.

    Args:
        past_moves (deque): A deque containing the FEN strings of past moves.
        current_board (chess.Board): The current chess board state.

    Returns:
        list: A list of dictionaries representing the system and user messages.
    """
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.format(
            past_moves="\n".join(
                "{}. {}".format(n, i) for n, i in enumerate(past_moves, start=1)
            ) if past_moves else "no past moves",
            current_move=current_board.fen()
        )}
    ]


def rl_make_move(current_board, past_moves):
    """
    Makes a move using the RL agent and logs the move duration to WandB.
    Returns:
        bool: True if the board state was reset due to too many invalid RL move generations, False otherwise.
    """
    start_time = time.time()
    retry = 0
    max_retries = 50

    while retry < max_retries:
        prompt = generate_prompt(past_moves, current_board)
        move_str = generate_move(prompt)
        try:
            move = chess.Move.from_uci(move_str)
            current_board.push(move)
            past_moves.append(current_board.fen())
            wandb.log({"rl_move_time": time.time() - start_time, "rl_move_number": len(past_moves)})
            return False
        except Exception:
            retry += 1
            print(f"Invalid move generated, retrying... ({retry}/{max_retries})")

    print("Too many retries, resetting last two moves.")
    for _ in range(2):
        current_board.pop()
        if past_moves:
            past_moves.pop()
    return True


def play_chess(engine="stockfish", side="random"):
    """
    Plays a series of chess games between the RL agent and the specified engine.
    """
    total_rl_wins, total_rl_draws, total_rl_losses = 0, 0, 0

    for game_count in range(1, config.max_games + 1):
        board = chess.Board()
        past_fen_moves = deque(maxlen=15)
        max_retry_count = 50
        retry_count = 0
        amount_moves = 0
        white_agent = None
        black_agent = None

        print(f"Starting game {game_count}")
        wandb.log({"game_id": game_count})

        is_rl_agent_white = {
            "random": random.choice([chess.WHITE, chess.BLACK]),
            "white": chess.WHITE,
            "black": chess.BLACK
        }.get(side, chess.WHITE)

        while not board.is_game_over():
            amount_moves += 1
            if retry_count >= max_retry_count:
                print("Too many retries, resetting game.")
                break

            original_turn = board.turn
            is_rl_turn = (original_turn and is_rl_agent_white) or (not original_turn and not is_rl_agent_white)

            if is_rl_turn:
                set_back = rl_make_move(board, past_fen_moves)
                agent = "RL Agent"
            else:
                set_back = engine_make_move(board, past_fen_moves, engine=engine)
                agent = "Stockfish Agent"

            if set_back:
                retry_count += 1
                amount_moves -= 2
                continue

            if original_turn:
                white_agent = agent
            else:
                black_agent = agent

        result = board.result()
        if result == "*":
            game_count -= 1
            print(f"Resetting game due to insufficient moves.")
            break

        print(f"Game {game_count} over: {result} with {amount_moves} moves. white: {white_agent}, black: {black_agent}")
        result_wandb = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}.get(result, 0)

        if result_wandb == 1:
            total_rl_wins += 1
        elif result_wandb == 0:
            total_rl_draws += 1
        elif result_wandb == -1:
            total_rl_losses += 1

        wandb.log({"game_result": result_wandb, "total_moves": amount_moves})

    print(
        f"Finished {config.max_games} games: RL wins: {total_rl_wins}, RL draws: {total_rl_draws}, "
        f"RL losses: {total_rl_losses}")
    wandb.log({
        "summary_rl_wins": total_rl_wins,
        "summary_rl_draws": total_rl_draws,
        "summary_rl_losses": total_rl_losses
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='JonasNasimzada/Llama-3.2-3B-Instruct', )
    parser.add_argument('--engine', choices=['stockfish', 'minmax'], default='stockfish', )
    parser.add_argument('--stockfish', type=str, default="../stockfish-ubuntu-x86-64-avx2", required=False,
                        help='Path to stockfish binary')
    parser.add_argument('--max_games', type=int, required=False, default=100)
    parser.add_argument('--side', choices=["random", "black", "white"], required=False, default="random")
    parser.add_argument('--wandb', type=str, required=False, default="chess_engine_evaluation")
    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"

    # WandB initialization
    wandb.init(
        project=args.wandb,
        config={
            "model_name": args.model,
            "stockfish_skill": 0,
            "stockfish_hash": 8,
            "stockfish_threads": 1,
            "max_games": args.max_games,
            "engine": args.engine,
            "side": args.side,
        }
    )
    config = wandb.config

    # Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )
    stockfish_agent = Stockfish(
        args.stockfish,
        depth=1,
        parameters={
            "Skill Level": config.stockfish_skill,
            "Debug Log File": f"./stockfish_debug_{args.model}_{args.engine}_{args.side}.log".replace("JonasNasimzada/",
                                                                                                      "").replace("/",
                                                                                                                  "_"),
            "Hash": config.stockfish_hash,
            "Threads": config.stockfish_threads,
        }
    )
    play_chess(engine=args.engine, side=args.side)
