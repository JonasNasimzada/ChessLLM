import copy
import re

import chess
import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from stockfish import Stockfish
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from utils import encoding

STOCKFISH_PATH = "/home/hk-project-pai00012/st_st171793/chessLLM/ChessLLM/stockfish/stockfish-ubuntu-x86-64-avx2 "
FEN_REGEX = r'^\s*^(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)\s([b|w])\s([K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$'
UCI_REGEX = r'^[a-h][1-8][a-h][1-8][nbrq]?$'


def isolate_fen_notation(prompt):
    user_prompt = prompt['messages'][1]["content"]
    search = re.search(FEN_REGEX, user_prompt)
    if search:
        fen = search.group(1)
        return fen
    else:
        return None


def isolate_move_notation(response):
    search = re.search(FEN_REGEX, response)
    if search:
        uci = search.group(1)
        return uci
    else:
        return None


def stockfish_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        engine = Stockfish(STOCKFISH_PATH)
        engine.set_skill_level(20)
        fen = isolate_fen_notation(prompt)
        engine.set_fen_position(fen)
        rating_before = engine.get_evaluation()['value']
        move = isolate_move_notation(completion)
        if not move:
            rewards.append(-5.0)
            continue
        engine.make_moves_from_current_position([move])
        rating_after = engine.get_evaluation()['value']
        reward = rating_after - rating_before
        rewards.append(reward)
    return rewards


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
            rewards.append(-5.0)
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
        try:
            move = chess.Move.from_uci(move_str)
            if move in chess_board.legal_moves:
                rewards.append(1.0)
            else:
                rewards.append(-5.0)
        except ValueError:
            rewards.append(-5.0)
    return rewards


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>[a-h][1-8][a-h][1-8][nbrq]?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else -1.0 for match in matches]


if __name__ == "__main__":
    dataset = load_dataset("./grpo_data/")  # Load your dataset here

    model_id = "JonasNasimzada/pretrained_chess_llm_ToC"
    device_string = PartialState().process_index

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={'': device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_tensor_parallel_size=1,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        beta=0.005,  # divergence coefficient
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        num_generations=4,
        temperature=0.5,
        max_prompt_length=2048,
        max_completion_length=512,
        num_train_epochs=2,
        logging_steps=100,
        save_steps=500,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="rl_chess_engine",
        push_to_hub=True,
        use_liger_kernel=True,

    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[stockfish_reward, end_game_reward, piece_reward, valid_uci_move_reward, format_reward_func],
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config
    )
    trainer.train()

    trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    trainer.model.config.use_cache = True
    trainer.save_model()
    training_args.distributed_state.wait_for_everyone()
    tokenizer.save_pretrained("rl_chess_engine")
    trainer.push_to_hub()
