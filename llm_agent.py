# llm_agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chess
import re
import random

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

# Load model and tokenizer; initial load may take time
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def get_move_by_llama(fen, max_try=3):
    """
    Get a legal UCI move for the given FEN using Llama LLM.
    """
    prompt = (
        f"Given the chess FEN: {fen}, what is the best move for the current player? "
        "Please answer only with the UCI format move, such as 'e2e4'."
    )
    for _ in range(max_try):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        move = extract_uci_move(reply)
        if move and is_legal_move(fen, move):
            return move
    # Return a random legal move if LLM fails
    board = chess.Board(fen)
    return next(iter(board.legal_moves)).uci()

def extract_uci_move(reply):
    """
    Extract UCI move (e.g. e2e4) from LLM output.
    """
    import re
    match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', reply)
    if match:
        return match.group(1)
    return None

def is_legal_move(fen, move_uci):
    """
    Check move legality under the given FEN.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except Exception:
        return False

def get_llm_board_eval(fen: str, max_try: int = 3) -> float:
    """
    Ask the LLM to evaluate a board position and return a float score.
    Positive ⇒ White is better; negative ⇒ Black is better (Stockfish-style).

    If the LLM fails to return a valid float within `max_try` attempts,
    fall back to a simple material evaluation.

    Parameters
    ----------
    fen : str
        Board position in FEN notation.
    max_try : int
        How many attempts to query the LLM before fallback.

    Returns
    -------
    float
        Board evaluation score.
    """
    prompt = (
        f"You are a chess engine. For the position below, "
        f"output a single evaluation number in centipawns "
        f"(positive means White is better, negative means Black is better). "
        f"Position (FEN): {fen}\n\nEvaluation:"
    )

    float_re = re.compile(r"-?\d+(\.\d+)?")

    for _ in range(max_try):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        m = float_re.search(reply)
        if m:
            try:
                return float(m.group())
            except ValueError:
                continue

    # ---------- fallback ----------
    board = chess.Board(fen)
    material = sum(
        (1, 3, 3, 5, 9, 0)[p.piece_type - 1] * (1 if p.color else -1)
        for p in board.piece_map().values()
    )
    # Add small random noise to avoid ties
    return material + random.uniform(-0.1, 0.1)

# Example usage
if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = get_move_by_llama(fen)
    print("LLM suggested move:", move)
