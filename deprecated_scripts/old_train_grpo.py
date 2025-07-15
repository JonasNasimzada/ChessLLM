####################################################################################
# THIS SCRIPT IS DEPRECATED AND NO LONGER USED. IT IS LEFT HERE FOR REFERENCE ONLY.#
####################################################################################

import copy
import re

import chess
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, setup_chat_format

from utils import encoding
from deprecated_scripts.stockfish import StockfishAgent

STOCKFISH_PATH = "/home/hk-project-pai00012/st_st171793/chessLLM/stockfish-ubuntu-x86-64-avx2"
FEN_REGEX = r'^\s*^(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)\s([b|w])\s([K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$'
UCI_REGEX = r'^[a-h][1-8][a-h][1-8][nbrq]?$'
# STOCKFISH = Stockfish(STOCKFISH_PATH)
# STOCKFISH.set_skill_level(20)

STOCKFISH = StockfishAgent(STOCKFISH_PATH)


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


def stockfish_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):

        fen = isolate_fen_notation(prompt)
        STOCKFISH.set_fen_position(fen)
        rating_before = STOCKFISH.get_evaluation()['value']
        move = isolate_move_notation(completion)
        if not move:
            rewards.append(-5.0)
            continue
        STOCKFISH.make_moves_from_current_position([move])
        rating_after = STOCKFISH.get_evaluation()['value']
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
            rewards.append(-1.0)
            continue
        move = chess.Move.from_uci(move_str)
        chess_board.push(move)
        lost, captured = encoding.evaluate_board_difference_score(old_board, chess_board)
        lost_count = -abs(sum(encoding.piece_reward[piece] for piece in lost) + 1)
        captured_count = sum(encoding.piece_reward[piece] for piece in captured)
        rewards.append(lost_count + captured_count)
    return rewards


def is_uci_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion is a valid UCI move."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        print(completion)
        move_str = isolate_move_notation(completion)
        if not move_str:
            rewards.append(-1.0)
            continue
        rewards.append(1.0)
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


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>[a-h][1-8][a-h][1-8][nbrq]?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else -1.0 for match in matches]


def load_pretrained_model():
    base_model_id = "openlm-research/open_llama_3b_v2"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "right"
    pre_trained_peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, pre_trained_peft_config)
    model = PeftModel.from_pretrained(
        lora_model,
        "JonasNasimzada/pretrained_chess_llm_ToC",
        from_tf=False,
        torch_dtype=torch.float16,
        device_map="auto",
        is_trainable=True,
    )
    return model, tokenizer, pre_trained_peft_config


if __name__ == "__main__":
    dataset = load_dataset("./grpo_data/")  # Load your dataset here

    pre_trained_model, pre_trained_tokenizer, peft_config = load_pretrained_model()
    pre_trained_model.unload()
    print("Pre-trained model loaded and unloaded successfully.")
    pre_trained_model, pre_trained_tokenizer, peft_config = load_pretrained_model()

    pre_trained_model, pre_trained_tokenizer = setup_chat_format(pre_trained_model, pre_trained_tokenizer)

    training_args = GRPOConfig(
        use_vllm=False,
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
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        num_generations=4,
        temperature=0.5,
        max_prompt_length=1548,
        max_completion_length=400,
        num_train_epochs=2,
        logging_steps=100,
        save_steps=500,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="rl_chess_engine3",
        push_to_hub=True,
        use_liger_kernel=False,

    )
    rl_peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    trainer = GRPOTrainer(
        model=pre_trained_model,
        processing_class=pre_trained_tokenizer,
        reward_funcs=[is_uci_reward],
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=rl_peft_config,
    )
    trainer.train()
    pre_trained_model.merge_and_unload()
    trainer.save_model()
    training_args.distributed_state.wait_for_everyone()
    pre_trained_tokenizer.save_pretrained("rl_chess_engine3")
    trainer.push_to_hub()
