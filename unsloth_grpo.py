import copy
import re

import chess
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from utils import encoding

FEN_REGEX = '^\s*^(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)\s([b|w])\s([K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$'
UCI_REGEX = r'[a-h][1-8][a-h][1-8][nbrq]?'


def isolate_fen_notation(prompt):
    user_prompt = prompt[1]["content"]
    search = re.findall(FEN_REGEX, user_prompt)
    if search:
        print("found fen")
        fen = search[-1]
        return fen
    else:
        return None


def isolate_move_notation(response):
    print(response)
    response = response[0]["content"]
    search = re.search(UCI_REGEX, response)
    if search:
        print("found uci")
        uci = search.group(1)
        return uci
    else:
        return None


def end_game_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt)
        chess_board = chess.Board(fen)
        board_turn = chess_board.turn
        move_str = isolate_move_notation(completion)
        if not move_str:
            rewards.append(-5.0)
            continue
        move = chess.Move.from_uci(move_str)
        chess_board.push(move)
        if chess_board.outcome():
            outcome = chess_board.outcome()
            if outcome.winner is not None:
                if outcome.winner == board_turn:
                    rewards.append(10.0)
                else:
                    rewards.append(-10.0)
    return rewards


def piece_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = isolate_fen_notation(prompt)
        chess_board = chess.Board(fen)
        old_board = copy.deepcopy(chess_board)
        move_str = isolate_move_notation(completion)
        if not move_str:
            rewards.append(-1.0)
            continue
        move = chess.Move.from_uci(move_str)
        chess_board.push(move)
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
        move_str = isolate_move_notation(completion)
        if not move_str:
            rewards.append(-1.0)
            continue
        try:
            move = chess.Move.from_uci(move_str)
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
            scores.append(-1)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            scores.append(2)
    return scores


if __name__ == "__main__":
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="JonasNasimzada/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    dataset = load_dataset("json", data_files="grpo_data/train_15_dataset.json", split="train")

    max_prompt_length = max(dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True, ).map(lambda x: {"length": len(x["tokens"])})["length"]) + 1

    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.20,
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=3,  # Set to 1 for a full training run
        save_steps=250,
        max_grad_norm=1.0,
        report_to="wandb",  # Can use Weights & Biases
        output_dir="outputs_unsloth/grpo",
        num_completions_to_print=2,
        log_completions=True,
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
    model.save_pretrained("llama_grpo")
    tokenizer.save_pretrained("llama_grpo")
    model.push_to_hub("llama_grpo")
    tokenizer.push_to_hub("llama_grpo")
