"""Train the PlasFlow v2 classifier (Random Forest + MLP).

Week 2 — Days 10–11 implementation target.

Usage:
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy
"""

from __future__ import annotations

import argparse
import logging
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
        import time

        from plasflow2.classify.train import evaluate, save_rf, split_data
        from plasflow2.classify.train import train_rf as _train_rf  # use library version
        from plasflow2.utils.device import IDX_TO_CLASS

        logger.info("Splitting data …")
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
        logger.info("Train=%d  Val=%d  Test=%d", len(X_tr), len(X_va), len(X_te))

        logger.info("Training Random Forest (500 trees, n_jobs=-1) …")
        t0 = time.time()
        rf = _train_rf(X_tr, y_tr, cv_folds=0)
        logger.info("RF trained in %.1f s", time.time() - t0)

        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]
        result = evaluate(y_te, rf.predict(X_te), class_names=class_names)
        logger.info("Test accuracy: %.4f", result["accuracy"])
        logger.info("\n%s", result["report"])

        save_rf(rf, out_dir / "rf_v2.pkl")

    if args.mlp:
        import time

        from plasflow2.classify.model import save_model
        from plasflow2.classify.train import evaluate, split_data
        from plasflow2.classify.train import train_mlp as _train_mlp
        from plasflow2.utils.device import IDX_TO_CLASS

        import gc

        logger.info("Splitting data …")
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
        logger.info("Train=%d  Val=%d  Test=%d", len(X_tr), len(X_va), len(X_te))

        # Free the full X and test arrays — not needed during training.
        # On macOS, keeping 4+ GB of numpy arrays alive alongside torch
        # tensors triggers memory pressure and a segfault.
        del X, y, X_te, y_te
        gc.collect()

        logger.info("Training MLP (AdamW + cosine LR + early stopping) …")
        t0 = time.time()
        model = _train_mlp(
            X_tr, y_tr, X_va, y_va,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            patience=10,
        )
        logger.info("MLP trained in %.1f s", time.time() - t0)

        import torch

        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]
        from plasflow2.utils.device import get_device

        device = get_device()
        with torch.no_grad():
            y_pred_mlp = (
                model(torch.tensor(X_te).float().to(device)).argmax(dim=-1).cpu().numpy()
            )
        result = evaluate(y_te, y_pred_mlp, class_names=class_names)
        logger.info("Test accuracy: %.4f", result["accuracy"])
        logger.info("\n%s", result["report"])

        save_model(model, out_dir / "mlp_v2.pt")


if __name__ == "__main__":
    main()
