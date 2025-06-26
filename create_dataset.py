import pandas as pd
from datasets import load_dataset, Dataset

system_message = """You are the best chest engine that plays chess games. As an input u get the current chess position postions of the past moves in FEN notation. You will generate the best chess move in UCI format. Here are the FEN notation of the past moves: {moves}"""

user_message = """Current chess position in FEN notation: {fen} - what is the next best move in UCI format?"""


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
    df = pd.DataFrame(dataset)
    df = df.sort_values(["game_index", "ply_index"])


    def add_context(group):
        fens = group["fen"].tolist()
        contexts = []
        for i in range(len(fens)):
            start = max(0, i - 27)
            # join the previous fens (i.e. positions before the current ply)
            before_fens = ", ".join(fens[start:i])
            if before_fens == '':
                contexts.append("no moves before")
            else:
                contexts.append(before_fens)
        group["context"] = contexts
        return group


    df = df.groupby("game_index", group_keys=False).apply(add_context)

    # df = df.sort_values(["game_index", "ply_index"], ignore_index=True)
    #
    # # 2) one‐pass loop to build contexts
    # contexts = []
    # current_game = None
    # history = []  # will hold up to 27 last fens for the *current* game
    #
    # for row in df.itertuples(index=False):
    #     game, fen = row.game_index, row.fen
    #
    #     # new game? reset your history buffer
    #     if game != current_game:
    #         current_game = game
    #         history.clear()
    #
    #     # build the context string
    #     if history:
    #         contexts.append(", ".join(history[-27:]))
    #     else:
    #         contexts.append("no moves before")
    #
    #     # append the current fen to history
    #     history.append(fen)
    #
    # # 3) assign back to your DataFrame
    # df["context"] = contexts

    print(df.head(3))
    ds = Dataset.from_pandas(df)

    # Convert dataset to OAI messages
    dataset = ds.map(instruction_format, remove_columns=dataset.features, batched=True, batch_size=1000)
    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)

    print(dataset["train"][345]["messages"])

    # save datasets to disk
    dataset["train"].to_json("./data/train/train_dataset.json", orient="records")
    dataset["test"].to_json("./data/test/test_dataset.json", orient="records")
