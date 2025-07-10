import pandas as pd
from datasets import load_dataset, Dataset

system_message = """You are the world’s strongest chess engine. You will be given the full move-history in FEN notation followed by the current position in FEN. Your task is to think through the position step by step—evaluating piece placement, pawn structure, king safety, candidate moves and tactical motifs—and then output exactly one best move in UCI format.\n\nStep-by-step guide:\n1. Material count and piece activity\n2. Pawn structure and central control\n3. King safety for both sides\n4. Candidate moves (e.g. developing, challenging the bishop, castling)\n5. Tactical considerations (pins, forks, discovered attacks)\n6. Long-term strategic plans\n\nAfter reasoning, output only the best move in UCI format.Respond in the following format:
<think>
You should reason between these tags.
</think>\n
The resulting UCI move should be between <answer> </answer> tags\n
Always use <think> </think> tags even if they are not necessary."""
user_message = """Move history (in FEN):\n{past_moves}\n\nCurrent position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""
user_message_no_context = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""


def instruction_format(sample):
    return {
        "prompt": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message.format(past_moves=sample["context"], current_move=sample["fen"])},
            # {"role": "user", "content": user_message_no_context.format(current_move=sample["fen"])},
        ],
        "answer": sample["move"]
    }


if __name__ == "__main__":
    # dataset = load_dataset("./trainings_data/", split="train")
    dataset = load_dataset("csv", data_files="trainings_data/train/test1.csv", split="train")
    ds = dataset.sort(["game_index", "ply_index"])

    contexts = []
    current_game = None
    prev_fens = []

    for ex in ds:
        gid = ex["game_index"]
        fen = ex["fen"]

        # if we hit a new game, reset history
        if gid != current_game:
            current_game = gid
            prev_fens = []

        # take up to the last 15 fens before the current move
        if len(prev_fens) == 0:
            contexts.append("no moves before")
        else:
            start = max(0, len(prev_fens) - 15)

            contexts.append("\n".join("{}. {}".format(n, i) for n, i in enumerate(prev_fens[start:], start=1)))

        # then record this move into history
        prev_fens.append(fen)

    # 3. Add the new column
    ds = ds.add_column("context", contexts)

    # Convert dataset to OAI messages
    dataset = ds.map(instruction_format, remove_columns=ds.column_names)

    # split dataset into 10,000 training samples and 2,500 test samples
    # dataset = dataset.train_test_split(train_size=0.9, test_size=0.1)

    # save datasets to disk
    dataset.to_json("./data/train_15_dataset.json", orient="records")
    # dataset["test"].to_json("./data/test_15_dataset.json", orient="records")
