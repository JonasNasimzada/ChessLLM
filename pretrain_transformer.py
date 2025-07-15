import argparse

import wandb
from pathlib import Path
import pandas as pd
import torch
import chess
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from utils.policyNetwork import SimpleTransformer

from utils.encoding import encode_board, encode_move

# ----------------------------------------------------------------------
# Configuration & W&B Initialization
# ----------------------------------------------------------------------

# Initialize a new W&B run
wandb.init(
    project="chess_policy_pretrain",
    config={
        "batch_size": 256,  # Number of samples per batch
        "chunksize": 15_000,  # Number of rows to process per chunk
        "epochs": 1000,  # Total number of training epochs
        "lr": 1e-4,  # Learning rate for the optimizer
        "model": "SimpleTransformer"  # Model name
    }
)
config = wandb.config


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class ChessDataset(Dataset):
    """
    A PyTorch Dataset for loading chess positions and moves.

    Args:
        df (pd.DataFrame): A DataFrame containing FEN strings and UCI moves.

    Returns:
        x (Tensor): A tensor of shape (960,) representing the board encoding with padding.
        y (Tensor): A scalar tensor representing the encoded move label.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: The board encoding and the encoded move label.
        """
        fen = self.df.iloc[idx]["fen"]
        uci = self.df.iloc[idx]["move"]
        board = chess.Board(fen)

        try:
            move_obj = chess.Move.from_uci(uci)
        except ValueError:
            return None, None

        board_vec = encode_board(board)  # Encode the board state (832,)
        zeros_128 = torch.zeros(128)  # Padding vector (128,)
        x = torch.cat([board_vec, zeros_128])  # Concatenate board encoding and padding (960,)
        y = encode_move(move_obj).argmax()  # Encode the move as a scalar tensor
        return x, y


def collate_fn(batch):
    """
    Custom collate function to filter out invalid samples.

    Args:
        batch (List[Tuple[Tensor, Tensor]]): A batch of samples.

    Returns:
        Tuple[Tensor, Tensor]: A batch of valid samples (inputs and labels).
    """
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None
    xb, yb = zip(*batch)
    return torch.stack(xb), torch.stack(yb)


# ----------------------------------------------------------------------
# Training helpers with W&B logging
# ----------------------------------------------------------------------

def train_one_chunk(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device,
                    epoch: int,
                    chunk_no: int) -> float:
    """
    Trains the model on a single chunk of data.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the current chunk.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to use for training.
        epoch (int): Current epoch number.
        chunk_no (int): Current chunk number.

    Returns:
        float: The average loss for the chunk.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        if batch is None:
            continue

        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        count += xb.size(0)

        # Log batch-level loss
        wandb.log({
            "batch_loss": loss.item(),
            "epoch": epoch,
            "chunk": chunk_no,
            "batch_idx": batch_idx
        })

    avg_loss = total_loss / count if count else 0.0
    # Log chunk-level average loss
    wandb.log({
        "chunk_avg_loss": avg_loss,
        "epoch": epoch,
        "chunk": chunk_no
    })
    return avg_loss


def train_from_large_csv(csv_file: str,
                         model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         loss_fn: nn.Module,
                         device: torch.device,
                         batch_size: int = None,
                         chunksize: int = None,
                         epochs: int = None) -> None:
    """
    Trains the model using data streamed from a large CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to use for training.
        batch_size (int, optional): Batch size for training. Defaults to config.batch_size.
        chunksize (int, optional): Number of rows to process per chunk. Defaults to config.chunksize.
        epochs (int, optional): Number of training epochs. Defaults to config.epochs.
    """
    # Update config to W&B
    batch_size = batch_size or config.batch_size
    chunksize = chunksize or config.chunksize
    epochs = epochs or config.epochs

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        for chunk_no, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize), start=1):
            print(f"\nChunk {chunk_no}: {len(chunk)} rows")
            ds = ChessDataset(chunk)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn,
            )

            avg_loss = train_one_chunk(model, dl, optimizer, loss_fn, device, epoch, chunk_no)
            print(f"Chunk {chunk_no} completed — average loss: {avg_loss:.4f}")

        if epoch % 20 == 0:
            ckpt_path = CHECKPOINT_DIR / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch} completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--ckpt_dir', type=str, required=False, default="checkpoints/pretrain_transformer/v2",
                        help="Directory to save model checkpoints.")
    args = parser.parse_args()

    csv_file = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path(args.ckpt_dir)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = SimpleTransformer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_from_large_csv(
        csv_file=csv_file,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_size=config.batch_size,
        chunksize=config.chunksize,
        epochs=config.epochs,
    )
    # Final save
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save(model.state_dict(), final_path)
    wandb.save(str(final_path))

    # Finish the run
    wandb.finish()
