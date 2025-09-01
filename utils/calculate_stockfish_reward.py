import math


def _norm_eval(ev, flipped):
    """
    Normalize Stockfish evaluation to the perspective of the original side to move.
    Stockfish evaluations are always from the perspective of the side to move.
    If it's the opponent's turn (flipped=True), we negate the score.
    """
    val = -ev["value"] if flipped else ev["value"]
    return {"type": ev["type"], "value": val}


def _bounded_reward_from_mate(baseline_v, after_v):
    """
    Compute a reward in [-1,1] based on mate distances.
    Positive mate values mean a forced mate in favor of the original side.
    Negative values mean the original side is being mated.
    - Winning mate: Closer mate = higher reward.
    - Losing mate: Delaying mate = higher reward.
    """
    if baseline_v > 0 and after_v > 0:
        # Both winning: smaller mate distance is better
        rb = 1.0 - math.tanh(abs(baseline_v) / 6.0)
        ra = 1.0 - math.tanh(abs(after_v) / 6.0)
        return ra - rb
    if baseline_v < 0 and after_v < 0:
        # Both losing: larger mate distance is better
        rb = -1.0 + math.tanh(abs(baseline_v) / 6.0)
        ra = -1.0 + math.tanh(abs(after_v) / 6.0)
        return ra - rb
    if baseline_v > 0 > after_v:
        # Went from winning to losing
        return -1.0
    if baseline_v < 0 < after_v:
        # Went from losing to winning
        return 1.0
    return 0.0


def _bounded_reward_from_cp(baseline_cp, after_cp, scale=300.0):
    """
    Compute reward from centipawn delta using a smooth tanh mapping.
    scale: CP value where tanh reaches ~0.76 (default 300 cp).
    """
    delta = after_cp - baseline_cp
    return math.tanh(delta / scale)


def evaluate_move_reward(fen, uci_move, engine, bounded=True):
    """
    Evaluate a move and return a scalar reward.
    Arguments:
      fen: FEN string of the current position.
      uci_move: The move to evaluate in UCI notation (e.g., "e2e4").
      engine: Stockfish binary.
      bounded: If True, return reward in [-1,1]. If False, return raw CP deltas and large constants for mates.

    Returns:
      A single float reward.
    """
    sf = engine
    if not sf.is_fen_valid(fen):
        return -1 if bounded else -10000.0

    # Baseline evaluation of current position
    sf.set_fen_position(fen)
    base_raw = sf.get_evaluation()
    base = _norm_eval(base_raw, flipped=False)

    # If move is illegal, return large negative reward
    if not sf.is_move_correct(uci_move):
        return -1.0 if bounded else -10000.0

    # Play the move
    sf.make_moves_from_current_position([uci_move])

    # Evaluation after the move (flip perspective)
    after_raw = sf.get_evaluation()
    after = _norm_eval(after_raw, flipped=True)

    # Mate-first reward calculation
    if base["type"] == "mate" or after["type"] == "mate":
        if base["type"] == "mate" and after["type"] == "mate":
            return _bounded_reward_from_mate(base["value"], after["value"])
        if base["type"] == "mate" and after["type"] != "mate":
            return -0.9
        return 0.9 if after["value"] > 0 else -0.9

    # CP-based reward
    if bounded:
        return _bounded_reward_from_cp(base["value"], after["value"])
    return float(after["value"] - base["value"])
