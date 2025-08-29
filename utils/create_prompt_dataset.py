import argparse

from datasets import load_dataset

# System message template for the chess engine's reasoning process
system_message = """Move history (in FEN):\n{past_moves}"""

# User message template for generating the next best move
user_message = """Current position (FEN):\n{current_move}\n\nWhat is the next best move in UCI format?"""

if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments, processes a dataset of chess games, 
    and generates a JSON dataset with instructions for training or evaluation.
    """

    # Parse command-line arguments
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
        help="Path to save the processed dataset in JSON format."
    )

    args = parser.parse_args()

    # Load the dataset from the specified CSV file
    dataset = load_dataset("csv", data_files=args.data, split="train")
    ds = dataset.sort(["game_index", "ply_index"])

    # Initialize variables for tracking game contexts
    contexts = []
    current_game = None
    prev_fens = []


    def instruction_format(sample):
        """
        Formats a dataset sample into an instruction dictionary for training or evaluation.

        Args:
            sample (dict): A dictionary containing the sample data, including FEN and move.

        Returns:
            dict: A formatted instruction dictionary with system and user messages, and the answer.
        """
        instruction = {}
        if args.type == "finetune":
            instruction = {
                "messages": [
                    {"role": "system", "content": system_message.format(past_moves=sample["context"])},
                    {"role": "user",
                     "content": user_message.format(current_move=sample["fen"])},
                    {"role": "assistant", "content": sample["move"]}
                ]
            }
        elif args.type == "grpo":
            instruction = {
                "prompt": [
                    {"role": "system", "content": system_message.format(past_moves=sample["context"])},
                    {"role": "user",
                     "content": user_message.format(current_move=sample["fen"])}
                ],
                "answer": sample["move"]
            }

        return instruction


    # Process each example in the dataset
    for ex in ds:
        gid = ex["game_index"]
        fen = ex["fen"]

        # If a new game is encountered, reset the move history
        if gid != current_game:
            current_game = gid
            prev_fens = []

        # Take up to the last 15 FENs before the current move
        if len(prev_fens) == 0:
            contexts.append("no moves before")
        else:
            start = max(0, len(prev_fens) - 15)
            contexts.append("\n".join("{}. {}".format(n, i) for n, i in enumerate(prev_fens[start:], start=1)))

        # Record the current move into the history
        prev_fens.append(fen)

    # Add the context column to the dataset
    ds = ds.add_column("context", contexts)

    # Map the dataset to the instruction format
    dataset = ds.map(instruction_format, remove_columns=ds.column_names, )

    # Save the processed dataset to a JSON file
    dataset.to_json(args.output, orient="records")
