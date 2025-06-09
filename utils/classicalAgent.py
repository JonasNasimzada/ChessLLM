import random

from utils.encoding import evaluate_material


#############################################
# Classical Agent (Minimax with Alpha-Beta)
#############################################

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_material(board)
    if maximizing:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval


def classical_agent_move(board, depth=3):
    legal_moves = list(board.legal_moves)
    best_moves = []
    best_eval = -float('inf')
    for move in legal_moves:
        board.push(move)
        current_eval = minimax(board, depth - 1, -float('inf'), float('inf'), False)
        board.pop()
        if current_eval > best_eval:
            best_eval = current_eval
            best_moves = [move]
        elif current_eval == best_eval:
            best_moves.append(move)
    if not best_moves:
        best_moves = legal_moves
    chosen_move = random.choice(best_moves)
    return chosen_move
