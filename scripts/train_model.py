"""Train the PlasFlow v2 classifier (Random Forest + MLP).

Week 2 — Days 10–11 implementation target.

Usage:
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def train_rf(X_train, y_train, n_estimators: int = 500):
    """Train Random Forest classifier.

    TODO (Day 10): implement full training with cross-validation.
    """
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def train_mlp(X_train, y_train, epochs: int = 50, batch_size: int = 512, lr: float = 1e-3):
    """Train PyTorch MLP on MPS/CUDA/CPU.

    TODO (Day 11): implement training loop with AdamW + cosine LR.
    """
    import torch
    from plasflow2.classify.model import PlasFlowMLP
    from plasflow2.utils.device import get_device
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device()
    model = PlasFlowMLP(input_dim=X_train.shape[1]).to(device)

    X_t = torch.tensor(X_train).float()
    y_t = torch.tensor(y_train).long()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, num_workers=0
    )  # num_workers=0 required for MPS

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

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
        if epoch % 10 == 0:
            logger.info("Epoch %d/%d — loss %.4f", epoch, epochs, total_loss / len(loader))

    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train PlasFlow v2 models")
    parser.add_argument("--data", required=True, help="Feature matrix (.npy)")
    parser.add_argument("--labels", required=True, help="Labels array (.npy)")
    parser.add_argument("--out", default="data/models", help="Output directory")
    parser.add_argument("--rf", action="store_true", help="Train Random Forest")
    parser.add_argument("--mlp", action="store_true", help="Train MLP")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.data).astype(np.float32)
    y = np.load(args.labels).astype(np.int64)
    logger.info("Loaded X=%s  y=%s", X.shape, y.shape)

    if args.rf:
        logger.info("Training Random Forest …")
        rf = train_rf(X, y)
        rf_path = out_dir / "rf_v2.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf, f)
        logger.info("Saved RF to %s", rf_path)

    if args.mlp:
        logger.info("Training MLP …")
        from plasflow2.classify.model import save_model

        model = train_mlp(X, y, epochs=args.epochs)
        save_model(model, out_dir / "mlp_v2.pt")


if __name__ == "__main__":
    main()
