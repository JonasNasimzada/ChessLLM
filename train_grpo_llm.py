"""
This script fine-tunes a pre-trained language model using GRPO (Guided Reward Policy Optimization) on a custom dataset.
It defines multiple reward functions for evaluating chess moves and trains the model to optimize these rewards.
The fine-tuned model is saved and optionally pushed to the Hugging Face model hub.
"""

import argparse
import copy
import re

from unsloth import FastLanguageModel
import chess
from datasets import load_dataset
from stockfish import Stockfish
from trl import GRPOConfig, GRPOTrainer
from vllm.config import CompilationConfig

from utils import encoding
from utils.calculate_stockfish_reward import evaluate_move_reward
from utils.encoding import isolate_fen_notation, isolate_move_notation, UCI_REGEX


def stockfish_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt[1]["content"])
        move_str = isolate_move_notation(completion[0]["content"])
        rating = evaluate_move_reward(fen, move_str, engine=STOCKFISH, bounded=True)
        rewards.append(rating)
    return rewards


def end_game_reward(prompts, completions, **kwargs):
    """
    Computes rewards based on the outcome of the chess game after a move.

    Args:
        prompts (list): List of prompts containing FEN strings.
        completions (list): List of completions containing move strings.

    Returns:
        list: Rewards for each move based on game outcome.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt[1]["content"])
        chess_board = chess.Board(fen)
        board_turn = chess_board.turn
        move_str = isolate_move_notation(completion[0]["content"])
        if not move_str:
            rewards.append(-5.0)
            continue
        try:
            move = chess.Move.from_uci(move_str)
        except chess.InvalidMoveError:
            rewards.append(-5.0)
            continue
        try:
            chess_board.push(move)
        except AssertionError:
            rewards.append(-5.0)
            continue
        if chess_board.outcome():
            outcome = chess_board.outcome()
            if outcome.winner is not None:
                if outcome.winner == board_turn:
                    rewards.append(10.0)
                else:
                    rewards.append(-10.0)
            else:
                rewards.append(-10.0)
        else:
            rewards.append(-1.0)
    return rewards


def piece_reward(prompts, completions, **kwargs):
    """
    Computes rewards based on the material gained or lost after a move.

    Args:
        prompts (list): List of prompts containing FEN strings.
        completions (list): List of completions containing move strings.

    Returns:
        list: Rewards for each move based on material changes.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt[1]["content"])
        chess_board = chess.Board(fen)
        old_board = copy.deepcopy(chess_board)
        move_str = isolate_move_notation(completion[0]["content"])
        if not move_str:
            rewards.append(-1.0)
            continue
        try:
            move = chess.Move.from_uci(move_str)
        except chess.InvalidMoveError:
            rewards.append(-5.0)
            continue
        try:
            chess_board.push(move)
        except AssertionError:
            rewards.append(-5.0)
            continue
        lost, captured = encoding.evaluate_board_difference_score(old_board, chess_board)
        lost_count = -abs(sum(encoding.piece_reward[piece] for piece in lost) + 1)
        captured_count = sum(encoding.piece_reward[piece] for piece in captured)
        rewards.append(lost_count + captured_count)
    return rewards


def valid_uci_move_reward(prompts, completions, **kwargs):
    """
    Computes rewards based on whether the move is a valid UCI move.

    Args:
        prompts (list): List of prompts containing FEN strings.
        completions (list): List of completions containing move strings.

    Returns:
        list: Rewards for each move based on validity.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt[1]["content"])
        chess_board = chess.Board(fen)
        move_str = isolate_move_notation(completion[0]["content"])
        if not move_str:
            rewards.append(-1.0)
            continue
        try:
            try:
                move = chess.Move.from_uci(move_str)
            except chess.InvalidMoveError:
                rewards.append(-1.0)
                continue
            if move in chess_board.legal_moves:
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        except ValueError:
            rewards.append(-1.0)
    return rewards


def check_answer(prompts, completions, answer, **kwargs):
    """
    Computes rewards based on whether the move matches the expected answer.

    Args:
        prompts (list): List of prompts containing FEN strings.
        completions (list): List of completions containing move strings.
        answer (list): List of correct answers.

    Returns:
        list: Rewards for each move based on correctness.
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := re.compile(UCI_REGEX).search(r)) is not None else None for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-1)
            continue
        if guess == true_answer:
            scores.append(1)
        else:
            scores.append(-1)
    return scores


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=True, default="JonasNasimzada/Llama-3.2-3B-Instruct",
                        help="Model which should be GRPO trained.")
    parser.add_argument('--dataset', type=str, required=True, default="grpo_data_dataset.json",
                        help="Path to the dataset file. File must be in JSON format with prompts and completions.")
    parser.add_argument('--output_model', type=str, required=True, default="grpo_model",
                        help="Name of the new model to be saved and pushed to the hub.")
    parser.add_argument('--output', type=str, default='output/grpo', required=False,
                        help="Output directory for the fine-tuned model.")
    parser.add_argument('--stockfish', type=str, default="../stockfish-ubuntu-x86-64-avx2", required=False,
                        help='Path to stockfish binary')
    parser.add_argument('--stockfish_skill', type=int, default=0, required=False, help='Skill level of Stockfish')

    args = parser.parse_args()

    # Model and tokenizer configuration
    max_seq_length = 2048
    lora_rank = 64

    comp_cfg = CompilationConfig(
        cache_dir="/hkfs/home/project/hk-project-pai00051/st_st171793/vllm_compile_cache",
        level=3,
        backend="inductor"
    )

    # Load pre-trained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.pretrained_model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        device_map='auto',
        compilation_config=comp_cfg,

    )

    # Apply PEFT (LoRA) to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Load and preprocess the dataset
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    max_prompt_length = max(dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True, ).map(lambda x: {"length": len(x["tokens"])})["length"]) + 1
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_seq_length - max_prompt_length}")

    # GRPO training configuration
    training_args = GRPOConfig(
        learning_rate=2e-4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=3,
        save_steps=250,
        max_grad_norm=1.0,
        report_to="wandb",
        output_dir=args.output,
        num_completions_to_print=1,
        log_completions=True,
        wandb_log_unique_prompts=True
    )

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            check_answer,
            # end_game_reward,
            # piece_reward,
            valid_uci_move_reward,
            stockfish_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    STOCKFISH = Stockfish(
        args.stockfish,
        parameters={
            "Skill Level": args.stockfish_skill,
        }
    )

    # Train the model
    trainer.train()

    # Save and push the fine-tuned model and tokenizer
    model.save_pretrained(args.output_model)
    tokenizer.save_pretrained(args.output_model)
    model.push_to_hub(args.output_model)
    tokenizer.push_to_hub(args.output_model)
