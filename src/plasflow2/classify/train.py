"""Training pipeline for PlasFlow v2 classifiers.

Day 3 implementation: ports v1's TensorFlow training logic to sklearn (RF)
and PyTorch (MLP) equivalents, with proper data splitting, cross-validation,
and evaluation metrics.

Week 2 target (Days 10–11): full training run on PLSDB/INPHARED/CARD data.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore[import]

logger = logging.getLogger(__name__)

# Reproducibility seed used throughout training
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


def split_data(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Stratified train / validation / test split.

    Args:
        X: Feature matrix, shape (N, D).
        y: Integer class labels, shape (N,).
        val_size: Fraction of total data held out for validation.
        test_size: Fraction of total data held out for final test.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First carve out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    # Then split the remainder into train + val
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        stratify=y_train_val,
        random_state=RANDOM_SEED,
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def train_rf(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    n_estimators: int = 500,
    cv_folds: int = 5,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with optional cross-validation reporting.

    This is the sklearn equivalent of PlasFlow v1's TensorFlow classifier,
    providing a fast, interpretable baseline.

    Args:
        X_train: Training feature matrix.
        y_train: Training class labels.
        n_estimators: Number of trees.
        cv_folds: Number of stratified CV folds for in-training evaluation.
                  Set to 0 to skip CV (faster, less diagnostic info).

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced",  # handles class imbalance between taxa
    )

    if cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_accs: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            rf_fold = RandomForestClassifier(
                n_estimators=100,  # fewer trees for CV speed
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
            rf_fold.fit(X_train[tr_idx], y_train[tr_idx])
            acc = accuracy_score(y_train[va_idx], rf_fold.predict(X_train[va_idx]))
            fold_accs.append(acc)
            logger.info("CV fold %d/%d — accuracy %.4f", fold, cv_folds, acc)
        logger.info("CV mean accuracy: %.4f ± %.4f", np.mean(fold_accs), np.std(fold_accs))

    # Train final model on the full training set
    rf.fit(X_train, y_train)
    logger.info("RF trained: %d trees, %d features", n_estimators, X_train.shape[1])
    return rf


def save_rf(rf: RandomForestClassifier, path: Path | str) -> None:
    """Pickle a trained Random Forest to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(rf, fh)
    logger.info("Saved RF to %s", path)


def load_rf(path: Path | str) -> RandomForestClassifier:
    """Load a pickled Random Forest from disk."""
    with open(path, "rb") as fh:
        rf = pickle.load(fh)  # noqa: S301
    logger.info("Loaded RF from %s", path)
    return rf


# ---------------------------------------------------------------------------
# MLP (PyTorch)
# ---------------------------------------------------------------------------


def train_mlp(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
) -> Any:  # returns PlasFlowMLP to avoid circular import at module level
    """Train the PyTorch MLP with early stopping on validation accuracy.

    Replaces PlasFlow v1's TensorFlow training loop with an AdamW + cosine LR
    schedule. MPS (Apple Silicon), CUDA, and CPU are all supported.

    Args:
        X_train: Training feature matrix (float32).
        y_train: Training class labels (int64).
        X_val: Validation feature matrix.
        y_val: Validation class labels.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size. Use ≤512 on MPS to avoid OOM.
        lr: Initial learning rate for AdamW.
        patience: Early-stopping patience (epochs without val improvement).

    Returns:
        PlasFlowMLP in eval mode, loaded with the best validation weights.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from plasflow2.classify.model import PlasFlowMLP
    from plasflow2.utils.device import get_device

    device = get_device()
    model = PlasFlowMLP(input_dim=X_train.shape[1]).to(device)

    X_t = torch.tensor(X_train).float()
    y_t = torch.tensor(y_train).long()
    X_v = torch.tensor(X_val).float().to(device)
    # y_val (numpy) is used for accuracy_score; X_v is the device tensor for inference

    # num_workers=0 required for MPS (fork-based multiprocessing not supported)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict[str, Any] = {}
    epochs_no_improve = 0

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

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_v)
            preds = logits.argmax(dim=-1).cpu().numpy()
        val_acc = accuracy_score(y_val, preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d — loss %.4f  val_acc %.4f  best %.4f",
                epoch,
                epochs,
                total_loss / len(loader),
                val_acc,
                best_val_acc,
            )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    logger.info("MLP training complete — best val accuracy: %.4f", best_val_acc)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, per-class F1, and confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Human-readable class names indexed by label value.
                     Only names for labels actually present in y_true are used,
                     so passing the full 4-class list is safe even when some
                     classes (e.g. archaea) are absent from the current dataset.

    Returns:
        Dict with keys: accuracy, report (str), confusion_matrix (ndarray).
    """
    acc = accuracy_score(y_true, y_pred)

    # Restrict target_names to labels that actually appear in the data
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if class_names is not None:
        present_names = [class_names[i] for i in present_labels if i < len(class_names)]
    else:
        present_names = None

    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    logger.info("Accuracy: %.4f\n%s", acc, report)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
