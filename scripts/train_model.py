"""Train the PlasFlow v2 classifier (Random Forest + MLP).

Usage:
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --mlp
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --rf

Memory design for MLP on macOS ARM
------------------------------------
With 400k × 1281 features the full array is ~2 GB.  Copying it into RAM
alongside torch tensors causes macOS memory pressure → segfault.

Solution: MmapDataset reads ONE BATCH at a time directly from the memory-
mapped .npy file.  The full training set is never in RAM simultaneously.

Peak RAM during training:
    Validation set:  40k × 1281 × 4 B  ≈  0.21 GB   (loaded once)
    One batch:      512 × 1281 × 4 B  ≈  0.003 GB  (ephemeral)
    Model weights:                     ≈  0.007 GB
    Total:                             ≈  0.22 GB   ← well within limits
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_SEED = 42


# ---------------------------------------------------------------------------
# Memory-mapped dataset — the core of the OOM fix
# ---------------------------------------------------------------------------


class MmapDataset(Dataset):
    """Read feature rows from a memory-mapped .npy file on demand.

    The file stays on disk; only the requested rows are paged in by the OS
    as each batch is constructed.  Works for arbitrarily large datasets.
    """

    def __init__(self, npy_path: str, indices: np.ndarray, labels: np.ndarray):
        # mmap_mode='r' → file-backed, not RAM-backed
        self._X = np.load(npy_path, mmap_mode="r")
        self._idx = indices     # which rows of the file belong to this split
        self._y = labels

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int):
        # .copy() copies one row (1281 × 4 B = 5 kB) from mmap → RAM
        x = torch.tensor(self._X[self._idx[i]].copy(), dtype=torch.float32)
        y = torch.tensor(self._y[i], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Training loop (standalone — does not call train.py's train_mlp)
# ---------------------------------------------------------------------------


def _train_mlp_mmap(
    data_path: str,
    idx_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
    out_path: Path = Path("data/models/mlp_v2.pt"),
) -> None:
    from sklearn.metrics import accuracy_score  # type: ignore[import]

    from plasflow2.classify.model import PlasFlowMLP, save_model
    from plasflow2.utils.device import get_device

    device = get_device()

    # Determine input_dim from the mmap without loading it fully
    X_mmap_meta = np.load(data_path, mmap_mode="r")
    input_dim = X_mmap_meta.shape[1]
    del X_mmap_meta

    model = PlasFlowMLP(input_dim=input_dim).to(device)
    logger.info("Model: input_dim=%d  device=%s", input_dim, device)

    # Validation tensor — 40k rows ≈ 0.21 GB, loaded once
    X_v = torch.tensor(X_va, dtype=torch.float32).to(device)
    logger.info("Validation tensor: %.2f GB on %s", X_v.nbytes / 1e9, device)

    # Training DataLoader — reads from mmap, one batch at a time
    train_ds = MmapDataset(data_path, idx_tr, y_tr)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,         # required — mmap + forked workers = bad
        pin_memory=False,
    )
    logger.info("Training batches per epoch: %d", len(loader))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict = {}
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=-1).cpu().numpy()
        val_acc = accuracy_score(y_va, preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d — loss %.4f  val_acc %.4f  best %.4f  [%.0f s]",
                epoch, epochs,
                total_loss / len(loader),
                val_acc, best_val_acc,
                elapsed,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
            break

    model.load_state_dict(best_state)
    model.eval()
    logger.info("Best validation accuracy: %.4f", best_val_acc)

    save_model(model, out_path)
    logger.info("Model saved → %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train PlasFlow v2 models")
    parser.add_argument("--data",   required=True, help="Feature matrix (.npy)")
    parser.add_argument("--labels", required=True, help="Labels array (.npy)")
    parser.add_argument("--out",    default="data/models", help="Output directory")
    parser.add_argument("--rf",     action="store_true", help="Train Random Forest")
    parser.add_argument("--mlp",    action="store_true", help="Train MLP")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Random Forest ────────────────────────────────────────────────────────
    if args.rf:
        from plasflow2.classify.train import evaluate, save_rf, split_data
        from plasflow2.classify.train import train_rf as _train_rf
        from plasflow2.utils.device import IDX_TO_CLASS

        X = np.load(args.data).astype(np.float32)
        y = np.load(args.labels).astype(np.int64)
        logger.info("Loaded X=%s  y=%s", X.shape, y.shape)

        X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
        logger.info("Train=%d  Val=%d  Test=%d", len(X_tr), len(X_va), len(X_te))

        t0 = time.time()
        rf = _train_rf(X_tr, y_tr, cv_folds=0)
        logger.info("RF trained in %.1f s", time.time() - t0)

        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]
        result = evaluate(y_te, rf.predict(X_te), class_names=class_names)
        logger.info("Test accuracy: %.4f", result["accuracy"])
        logger.info("\n%s", result["report"])
        save_rf(rf, out_dir / "rf_v2.pkl")

    # ── MLP — mmap-based training, never loads full X into RAM ───────────────
    if args.mlp:
        from sklearn.model_selection import train_test_split  # type: ignore[import]

        # Step 1: split INDICES only (labels are 3 MB — trivial)
        logger.info("Loading labels and splitting indices …")
        y_all = np.load(args.labels).astype(np.int64)
        n = len(y_all)
        logger.info("Total samples: %d", n)

        idx_all = np.arange(n)
        idx_trainval, idx_te, y_trainval, _ = train_test_split(
            idx_all, y_all, test_size=0.10, stratify=y_all, random_state=_SEED,
        )
        idx_tr, idx_va, y_tr, y_va = train_test_split(
            idx_trainval, y_trainval,
            test_size=0.10 / 0.90,
            stratify=y_trainval,
            random_state=_SEED,
        )
        del idx_all, y_all, idx_trainval, y_trainval, idx_te
        gc.collect()
        logger.info("Split: Train=%d  Val=%d", len(idx_tr), len(idx_va))

        # Step 2: load ONLY the validation slice into RAM (~0.21 GB)
        logger.info("Loading validation slice into RAM …")
        X_mmap = np.load(args.data, mmap_mode="r")
        X_va_np = np.ascontiguousarray(X_mmap[idx_va]).astype(np.float32)
        del X_mmap
        gc.collect()
        logger.info("X_va in RAM: %.2f GB  (training data stays on disk)", X_va_np.nbytes / 1e9)

        # Step 3: train using MmapDataset — batches read from disk on demand
        _train_mlp_mmap(
            data_path=args.data,
            idx_tr=idx_tr,
            y_tr=y_tr,
            X_va=X_va_np,
            y_va=y_va,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            patience=10,
            out_path=out_dir / "mlp_v2.pt",
        )


if __name__ == "__main__":
    main()
