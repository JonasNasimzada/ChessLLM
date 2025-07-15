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
        "batch_size": 256,
        "chunksize": 15_000,
        "epochs": 1000,
        "lr": 1e-4,
        "model": "SimpleTransformer"
    }
)
config = wandb.config


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class ChessDataset(Dataset):
    """
    Returns:
        x : Tensor, shape (960,)   – board encoding + 128 zero-padding
        y : Tensor, shape (1,)     – scalar 1.0  (expert move label)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        fen = self.df.iloc[idx]["fen"]
        uci = self.df.iloc[idx]["move"]
        board = chess.Board(fen)

        try:
            move_obj = chess.Move.from_uci(uci)
        except ValueError:
            return None, None

        board_vec = encode_board(board)  # (832,)
        zeros_128 = torch.zeros(128)  # (128,)
        x = torch.cat([board_vec, zeros_128])  # (960,)
        y = encode_move(move_obj).argmax()  # scalar tensor
        return x, y


def collate_fn(batch):
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
    """Train on a single DataLoader chunk and return the average loss."""
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
    """Stream a CSV file in manageable chunks and train the model."""
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
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=False, default="checkpoints/pretrain_transformer/v2")
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
    # final save
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save(model.state_dict(), final_path)
    wandb.save(str(final_path))

    # Finish the run
    wandb.finish()
