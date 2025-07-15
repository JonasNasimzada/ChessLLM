import argparse
import copy
import re

import chess
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from utils import encoding
from utils.encoding import isolate_fen_notation, isolate_move_notation, UCI_REGEX


def end_game_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt)
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
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt)
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
    """Reward function that checks if the completion is a valid UCI move."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt)
        chess_board = chess.Board(fen)
        move_str = isolate_move_notation(completion[0]["content"])
        if not move_str:
            rewards.append(-1.0)
            continue
        try:
            try:
                move = chess.Move.from_uci(move_str)
            except chess.InvalidMoveError:
                rewards.append(-5.0)
                continue
            if move in chess_board.legal_moves:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except ValueError:
            rewards.append(-1.0)
    return rewards


def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := re.compile(UCI_REGEX).search(r)) is not None else None for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2)
            continue
        if guess == true_answer:
            scores.append(2)
        else:
            scores.append(-1)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=True, default="JonasNasimzada/Llama-3.2-3B-Instruct",
                        help="Model which should be grpo trained")
    parser.add_argument('--dataset', type=str, required=True, default="grpo_data_dataset.json",
                        help="Path to the dataset file. file has to be json format with prompts and completions.")
    parser.add_argument('--output_model', type=str, required=True, default="grpo_model",
                        help="Name of the new model to be saved and pushed to the hub.")
    parser.add_argument('--output', type=str, default='output/grpo', required=False,
                        help="Output directory for the fine-tuned model. ")

    args = parser.parse_args()

    max_seq_length = 2048
    lora_rank = 64

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.pretrained_model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        device_map='auto',

    )

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

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    max_prompt_length = max(dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True, ).map(lambda x: {"length": len(x["tokens"])})["length"]) + 1
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_seq_length - max_prompt_length}")

    training_args = GRPOConfig(
        learning_rate=5e-6,
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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            check_answer,
            end_game_reward,
            piece_reward,
            valid_uci_move_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output_model)
    tokenizer.save_pretrained(args.output_model)
    model.push_to_hub(args.output_model)
    tokenizer.push_to_hub(args.output_model)
