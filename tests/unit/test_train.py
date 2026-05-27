"""Unit tests for the Day 3 training pipeline (classify/train.py).

These tests use small synthetic datasets so they run without real genomic data
and complete in < 5 seconds even on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from plasflow2.classify.train import (
    RANDOM_SEED,
    evaluate,
    load_rf,
    save_rf,
    split_data,
    train_rf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(RANDOM_SEED)


def _make_dummy(n: int = 200, d: int = 64, n_classes: int = 4) -> tuple:
    """Return a small random feature matrix and balanced label vector."""
    X = RNG.standard_normal((n, d)).astype(np.float32)
    y = np.repeat(np.arange(n_classes, dtype=np.int64), n // n_classes)
    return X, y


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


def test_split_sizes() -> None:
    X, y = _make_dummy(200)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
    total = len(X_tr) + len(X_va) + len(X_te)
    assert total == 200


def test_split_no_overlap() -> None:
    X, y = _make_dummy(200)
    # Use indices: assign each row a unique id via a column of sequential ints
    X_idx = np.hstack([X, np.arange(200).reshape(-1, 1).astype(np.float32)])
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X_idx, y, val_size=0.1, test_size=0.1)
    ids_tr = set(X_tr[:, -1].astype(int))
    ids_va = set(X_va[:, -1].astype(int))
    ids_te = set(X_te[:, -1].astype(int))
    assert not ids_tr & ids_va, "Train and val overlap"
    assert not ids_tr & ids_te, "Train and test overlap"
    assert not ids_va & ids_te, "Val and test overlap"


def test_split_stratified() -> None:
    """Each split should contain samples from all classes."""
    X, y = _make_dummy(200)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
    for part_y, name in [(y_tr, "train"), (y_va, "val"), (y_te, "test")]:
        classes_in_split = set(part_y.tolist())
        assert len(classes_in_split) == 4, f"{name} split missing classes: {classes_in_split}"


# ---------------------------------------------------------------------------
# train_rf
# ---------------------------------------------------------------------------


def test_rf_trains_and_predicts() -> None:
    X, y = _make_dummy(200)
    X_tr, _, _, y_tr, _, _ = split_data(X, y)
    rf = train_rf(X_tr, y_tr, n_estimators=10, cv_folds=0)
    preds = rf.predict(X_tr)
    assert preds.shape == y_tr.shape
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_rf_accuracy_above_chance() -> None:
    """RF should beat random guessing (25%) even on tiny synthetic data."""
    X, y = _make_dummy(200)
    X_tr, _, X_te, y_tr, _, y_te = split_data(X, y)
    rf = train_rf(X_tr, y_tr, n_estimators=20, cv_folds=0)
    acc = (rf.predict(X_te) == y_te).mean()
    assert acc > 0.25, f"RF accuracy {acc:.2f} is not above random chance"


def test_rf_cv_runs_without_error() -> None:
    X, y = _make_dummy(200)
    X_tr, _, _, y_tr, _, _ = split_data(X, y)
    # Should complete without raising; cv_folds=3 for speed
    train_rf(X_tr, y_tr, n_estimators=5, cv_folds=3)


def test_rf_save_load_roundtrip(tmp_path) -> None:
    X, y = _make_dummy(200)
    X_tr, _, _, y_tr, _, _ = split_data(X, y)
    rf = train_rf(X_tr, y_tr, n_estimators=10, cv_folds=0)
    preds_before = rf.predict(X_tr)

    path = tmp_path / "rf_test.pkl"
    save_rf(rf, path)
    rf2 = load_rf(path)
    preds_after = rf2.predict(X_tr)

    np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    result = evaluate(y, y, class_names=["plasmid", "chromosome", "phage", "archaea"])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"].trace() == len(y)


def test_evaluate_returns_required_keys() -> None:
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    result = evaluate(y, y)
    assert "accuracy" in result
    assert "report" in result
    assert "confusion_matrix" in result


def test_evaluate_partial_accuracy() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)  # 3/4 correct
    result = evaluate(y_true, y_pred)
    assert result["accuracy"] == pytest.approx(0.75)
