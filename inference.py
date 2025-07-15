import os
import random
import time
from collections import deque
import sys

import chess
import torch
import wandb
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from stockfish import Stockfish

from utils.classicalAgent import ClassicalAgent
from utils.stockfish import StockfishAgent

from utils.encoding import isolate_move_notation

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

system_message = """You are the world’s strongest chess engine. You will be given the full move-history in FEN notation followed by the current position in FEN. Your task is to think through the position step by step—evaluating piece placement, pawn structure, king safety, candidate moves and tactical motifs—and then output exactly one best move in UCI format.\n\nStep-by-step guide:\n1. Material count and piece activity\n2. Pawn structure and central control\n3. King safety for both sides\n4. Candidate moves (e.g. developing, challenging the bishop, castling)\n5. Tactical considerations (pins, forks, discovered attacks)\n6. Long-term strategic plans\n\nAfter reasoning, output only the best move in UCI format.Respond in the following format:
<think>
You should reason between these tags.
</think>\n
The resulting UCI move should be between <answer> </answer> tags\n
Always use <think> </think> tags even if they are not necessary."""

user_message = """Move history (in FEN):\n{past_moves}\n\nCurrent position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""
user_message_no_context = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""


def engine_make_move(current_board, past_fen_moves, engine="stockfish"):
    set_moves_back = False
    if engine == "stockfish":
        if not current_board.is_valid():
            print("Invalid board state, resetting last two moves.")
            for i in range(2):
                current_board.pop()
                past_fen_moves.pop()
            set_moves_back = True

        fen = current_board.fen()
        past_fen_moves.append(fen)
        stockfish_agent.set_fen_position(fen)
        move = stockfish_agent.get_best_move()
        move = chess.Move.from_uci(move)

    elif engine == "minmax":
        engine = ClassicalAgent(depth=3)
        move = engine.get_move(board=current_board)

    current_board.push(move)
    return set_moves_back


def generate_move(prompt):
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


def rl_make_move(current_board, past_moves):
    start_time = time.time()
    current_fen = current_board.fen()
    prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.format(
            past_moves="\n".join(
                "{}. {}".format(n, i) for n, i in enumerate(past_moves, start=1)
            ) if past_moves else "no past moves",
            current_move=current_fen
        )},
    ]

    move_str = generate_move(prompt)
    while True:
        try:
            move = chess.Move.from_uci(move_str)
            current_board.push(move)
            past_moves.append(current_board.fen())
            break
        except Exception:
            move_str = generate_move(prompt)
            print("Invalid move generated, retrying...")

    duration = time.time() - start_time
    wandb.log({
        "move_time_rl": duration,
        "move_number": len(past_moves)
    })


def play_chess(engine="stockfish"):
    game_count = 0
    total_rl_wins = 0
    total_rl_draws = 0
    total_rl_losses = 0

    board = chess.Board()
    while game_count < config.max_games:
        board.reset()
        game_count += 1
        amount_moves = 0
        past_fen_moves = deque(maxlen=15)
        white_agent = None
        black_agent = None
        print(f"Starting game {game_count}")

        # Log game start
        wandb.log({"game_id": game_count})

        is_rl_agent_white = chess.WHITE if random.choice([True, False]) else chess.BLACK
        retry_count = 0
        while not board.is_game_over():
            amount_moves += 1
            if retry_count == 70:
                print("Too many retries, resetting game.")
                break
            if board.turn:  # White's turn
                if is_rl_agent_white:
                    rl_make_move(board, past_fen_moves)
                    white_agent = "RL Agent"
                else:
                    set_back = engine_make_move(board, past_fen_moves, engine=engine)
                    if set_back:
                        retry_count += 1
                        amount_moves -= 2
                    white_agent = "Stockfish Agent"

            else:  # Black's turn
                if not is_rl_agent_white:
                    rl_make_move(board, past_fen_moves)
                    black_agent = "RL Agent"
                else:
                    set_back = engine_make_move(board, past_fen_moves, engine=engine)
                    if set_back:
                        retry_count += 1
                        amount_moves -= 2
                    black_agent = "Stockfish Agent"

        result = board.result()
        print(
            f"Game {game_count} over: {result} with {amount_moves} moves. white: {white_agent}, black: {black_agent}"
        )
        result_wandb = 0

        # Track RL agent wins
        if (result == "1-0" and is_rl_agent_white) or (result == "0-1" and not is_rl_agent_white):
            total_rl_wins += 1
            result_wandb = 1
        # Track RL agent draws
        if result == "1/2-1/2":
            total_rl_draws += 1
            result_wandb = 0
        # Track RL agent losses
        if (result == "0-1" and is_rl_agent_white) or (result == "1-0" and not is_rl_agent_white):
            total_rl_losses += 1
            result_wandb = -1

        # Log game results
        wandb.log({
            "game_result": result_wandb,
            "total_moves": amount_moves,
        })
    # Print summary statistics
    print(
        f"Finished {config.max_games} games: "
        f"RL wins: {total_rl_wins}, "
        f"RL draws: {total_rl_draws}, "
        f"RL losses: {total_rl_losses}"
    )
    # Log summary stats to WandB
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
    args = parser.parse_args()

    log_file = f'inference_{args.model}_{args.engine}.log'.replace("JonasNasimzada/", "")
    sys.stdout = open(log_file, 'w')

    # WandB initialization
    wandb.init(
        project="chess_engine_evaluation",
        config={
            "model_name": args.model,
            "stockfish_skill": 0,
            "stockfish_hash": 2048,
            "stockfish_threads": 1,
            "max_games": 100,
            "engine": args.engine,
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
        "../stockfish-ubuntu-x86-64-avx2",
        parameters={
            "Skill Level": config.stockfish_skill,
            "Debug Log File": f"./stockfish_debug_{args.model}_{args.engine}.log",
            "Hash": config.stockfish_hash,
            "Threads": config.stockfish_threads,
        }
    )
    play_chess(engine=args.engine)
