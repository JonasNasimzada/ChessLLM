import pandas as pd
from datasets import load_dataset, Dataset

system_message = """You are the best chest engine that plays chess games. As an input u get the current chess position postions of the past moves in FEN notation. You will generate the best chess move in UCI format. Here are the FEN notation of the past moves: {moves}"""

user_message = """Current chess position in FEN notation: {fen} {{what is the next best move in UCI format?}}"""


def instruction_format(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message.format(moves=sample["context"])},
            {"role": "user", "content": user_message.format(fen=sample["fen"])},
            {"role": "assistant", "content": sample["move"]}
        ]
    }


if __name__ == "__main__":
    dataset = load_dataset("./trainings_data/", split="train")
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

        # take up to the last 27 fens before the current move
        if len(prev_fens) == 0:
            contexts.append("no moves before")
        else:
            start = max(0, len(prev_fens) - 27)
            contexts.append(", ".join(prev_fens[start:]))

        # then record this move into history
        prev_fens.append(fen)

    # 3. Add the new column
    ds = ds.add_column("context", contexts)

    # Convert dataset to OAI messages
    dataset = ds.map(instruction_format, remove_columns=ds.column_names)

    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)

    # save datasets to disk
    dataset["train"].to_json("./data/train/train_dataset.json", orient="records")
    dataset["test"].to_json("./data/test/test_dataset.json", orient="records")
