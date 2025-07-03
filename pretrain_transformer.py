"""
Train a policy network on a very large CSV of <FEN, move> pairs.
The file is streamed in chunks to keep RAM usage under control.
"""

from pathlib import Path
import pandas as pd
import torch
import chess
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

from utils.encoding import encode_board, encode_move

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CHECKPOINT_DIR = Path("checkpoints/pretrain_transformer")
CHECKPOINT_DIR.mkdir(exist_ok=True)


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
        san = self.df.iloc[idx]["move"]
        board = chess.Board(fen)

        # Ignore rows with malformed SAN moves
        try:
            move_obj = board.parse_san(san)
        except ValueError:
            return None, None

        # Input vector: board encoding + 128 zeros = 960‑dim
        board_vec = encode_board(board)  # (832,)
        zeros_128 = torch.zeros(128)  # (128,)
        x = torch.cat([board_vec, zeros_128])  # (960,)
        # Target index 0‥127  (class label for CrossEntropyLoss)
        y = encode_move(move_obj).argmax()  # scalar tensor
        return x, y


def collate_fn(batch):
    """Remove invalid samples and return `None` if an entire batch is invalid."""
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None
    xb, yb = zip(*batch)
    return torch.stack(xb), torch.stack(yb)


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------

def train_one_chunk(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device) -> float:
    """Train on a single DataLoader chunk and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None:
            continue  # entire batch was invalid

        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) else 0.0


def train_from_large_csv(csv_file: str,
                         model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         loss_fn: nn.Module,
                         device: torch.device,
                         batch_size: int = 128,
                         chunksize: int = 10_000,
                         epochs: int = 1) -> None:
    """Stream a CSV file in manageable chunks and train the model."""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
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

            avg_loss = train_one_chunk(model, dl, optimizer, loss_fn, device)
            print(f"Chunk {chunk_no} completed — average loss: {avg_loss:.4f}")

            ckpt_path = CHECKPOINT_DIR / f"epoch{epoch}_chunk{chunk_no}.pt"
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch + 1} completed.\n")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    csv_file = "LumbrasGigaBase 2024.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from policyNetwork import SimpleTransformer

    model = SimpleTransformer().to(device)  # 960‑in, 1‑out
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_from_large_csv(
        csv_file=csv_file,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_size=256,  # decrease if GPU memory is limited
        chunksize=10_000,  # decrease if system RAM is limited
        epochs=1,
    )
    # final save
    torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
