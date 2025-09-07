#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset

# ----------------------------
# Templates
# ----------------------------
system_message = """Move history (in FEN):\n{past_moves}"""
user_message = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""

def init_distributed_if_needed() -> dict:
    """
    Initialize torch.distributed if launched with torchrun.
    Returns a dict with rank, world_size, local_rank, and a boolean 'distributed'.
    """
    env = os.environ
    distributed = "WORLD_SIZE" in env and int(env["WORLD_SIZE"]) > 1
    rank = int(env.get("RANK", "0"))
    world_size = int(env.get("WORLD_SIZE", "1"))
    local_rank = int(env.get("LOCAL_RANK", "0"))

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }

def barrier(enabled: bool):
    if enabled and dist.is_initialized():
        dist.barrier()

def write_jsonl(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def main():
    """
    Entry point. Shards work across GPUs (via torchrun), processes the CSV with Hugging Face Datasets,
    and writes per-rank JSONL shards which rank 0 merges into the final JSON (records array).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="finetune",
        choices=["finetune", "grpo"],
        required=False,
        help="Type of extraction to perform (finetune or grpo)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="LumbrasGigaBase_OTB_2025.csv",
        required=True,
        help="Path to the input CSV file containing chess game data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_dataset.json",
        help="Path to save the merged dataset (JSON array of records)."
    )
    parser.add_argument(
        "--past_moves",
        type=int,
        default=15,
        help="Number of past moves (FENs) to include in the context."
    )
    args = parser.parse_args()

    ddp = init_distributed_if_needed()
    distributed = ddp["distributed"]
    rank = ddp["rank"]
    world_size = ddp["world_size"]

    # Load and sort once, then shard across workers
    dataset = load_dataset("csv", data_files=args.data, split="train")
    ds = dataset.sort(["game_index", "ply_index"])

    # Shard by sample to avoid overlap
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)

    # Build contexts per (sharded) game
    contexts = []
    current_game = None
    prev_fens: List[str] = []

    for ex in ds:
        gid = ex["game_index"]
        fen = ex["fen"]

        if gid != current_game:
            current_game = gid
            prev_fens = []

        if len(prev_fens) == 0:
            contexts.append("no moves before")
        else:
            start = max(0, len(prev_fens) - args.past_moves)
            contexts.append("\n".join(f"{n}. {i}" for n, i in enumerate(prev_fens[start:], start=1)))

        prev_fens.append(fen)

    ds = ds.add_column("context", contexts)

    def instruction_format(sample):
        if args.type == "finetune":
            return {
                "messages": [
                    {"role": "system", "content": system_message.format(past_moves=sample["context"])},
                    {"role": "user", "content": user_message.format(current_move=sample["fen"])},
                    {"role": "assistant", "content": sample["move"]}
                ]
            }
        else:  # "grpo"
            return {
                "prompt": [
                    {"role": "system", "content": system_message.format(past_moves=sample["context"])},
                    {"role": "user", "content": user_message.format(current_move=sample["fen"])}
                ],
                "answer": sample["move"]
            }

    # Transform to instruction records
    transformed = ds.map(instruction_format, remove_columns=ds.column_names)

    # Write per-rank JSONL shard (easier to merge)
    shard_path = f"{args.output}.rank{rank}.jsonl" if world_size > 1 else f"{args.output}.jsonl"
    write_jsonl(shard_path, transformed.to_list())

    # Sync all ranks before merging
    barrier(distributed)

    # Rank 0 merges shards into a single JSON array file (args.output)
    if rank == 0:
        all_records: List[dict] = []
        if world_size > 1:
            for r in range(world_size):
                p = f"{args.output}.rank{r}.jsonl"
                if not os.path.exists(p):
                    # If a shard is absent (empty slice), skip
                    continue
                all_records.extend(read_jsonl(p))
        else:
            all_records.extend(read_jsonl(shard_path))

        # Final output: JSON array of records
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False)

        # Optional: clean up shard files
        for r in range(world_size if world_size > 1 else 1):
            try:
                os.remove(f"{args.output}.rank{r}.jsonl" if world_size > 1 else shard_path)
            except OSError:
                pass

    # Final sync so non-zero ranks don't exit before merge completes
    barrier(distributed)

    # Graceful teardown
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()