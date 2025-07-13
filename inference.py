import random
import time
from collections import deque

import chess
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from stockfish import Stockfish
from utils.stockfish import StockfishAgent

from utils.encoding import isolate_move_notation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

system_message = """You are the world’s strongest chess engine. You will be given the full move-history in FEN notation followed by the current position in FEN. Your task is to think through the position step by step—evaluating piece placement, pawn structure, king safety, candidate moves and tactical motifs—and then output exactly one best move in UCI format.\n\nStep-by-step guide:\n1. Material count and piece activity\n2. Pawn structure and central control\n3. King safety for both sides\n4. Candidate moves (e.g. developing, challenging the bishop, castling)\n5. Tactical considerations (pins, forks, discovered attacks)\n6. Long-term strategic plans\n\nAfter reasoning, output only the best move in UCI format.Respond in the following format:
<think>
You should reason between these tags.
</think>\n
The resulting UCI move should be between <answer> </answer> tags\n
Always use <think> </think> tags even if they are not necessary."""
user_message = """Move history (in FEN):\n{past_moves}\n\nCurrent position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""
user_message_no_context = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""


def stockfish_make_move(current_board):
    # fen = current_board.fen()
    # stockfish_engine.set_fen_position(fen)
    # result = stockfish_engine.get_best_move()
    time_limit = 1.0  # seconds
    while True:
        try:
            move = stockfish_agent.get_move(current_board, time_limit=time_limit, ponder=True)
            break
        except Exception:
            time_limit += 0.5  # Increase time limit if Stockfish fails to generate a move
            print("Stockfish failed to generate a move, increasing time limit to", time_limit)
    # move = chess.Move.from_uci(result)
    current_board.push(move)


def generate_move(prompt):
    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(
        DEVICE)
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=1024, use_cache=True,
                            temperature=1.5, min_p=0.1)
    move_str = tokenizer.decode(output[0], skip_special_tokens=True)
    move = isolate_move_notation(move_str)
    return move


def rl_make_move(current_board, past_moves):
    current_fen = current_board.fen()
    prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.format(
            past_moves="\n".join(
                "{}. {}".format(n, i) for n, i in enumerate(past_moves, start=1)) if past_moves else "no past moves",
            current_move=current_fen)},
    ]

    move_str = generate_move(prompt)
    while True:
        try:
            move = chess.Move.from_uci(move_str)
            current_board.push(move)
            past_moves.append(current_board.fen())
            break
        except chess.InvalidMoveError or ValueError:
            move_str = generate_move(prompt)
            print("Invalid move generated, retrying...")


def play_chess():
    game_count = 0
    board = chess.Board()
    while game_count < 2:
        board.reset()
        game_count += 1
        amount_moves = 0
        past_fen_moves = deque(maxlen=15)
        white_agent = None
        black_agent = None
        print(f"Starting game {game_count}")

        is_rl_agent_white = chess.WHITE if random.choice([True, False]) else chess.BLACK

        while not board.is_game_over():
            amount_moves += 1
            if board.turn:  # White.
                if is_rl_agent_white:
                    rl_make_move(board, past_fen_moves)
                    white_agent = "RL Agent"
                else:
                    stockfish_make_move(board)
                    white_agent = "Stockfish Agent"

            else:  # Black.
                if not is_rl_agent_white:
                    rl_make_move(board, past_fen_moves)
                    black_agent = "RL Agent"
                else:
                    stockfish_make_move(board)
                    black_agent = "Stockfish Agent"
        print(
            f"Game {game_count} over: {board.result()} with {amount_moves} moves. white: {white_agent}, black: {black_agent}")

        result = board.result()


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="JonasNasimzada/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )
    # stockfish_engine = Stockfish("../stockfish-ubuntu-x86-64-avx2")
    # stockfish_engine.set_skill_level(0)
    stockfish_agent = StockfishAgent(stockfish_path="../stockfish-ubuntu-x86-64-avx2", config={"Skill Level": 0})
    play_chess()
