"""Train the PlasFlow v2 classifier (Random Forest + MLP).

Usage:
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --mlp
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --rf

Memory strategy for MLP on macOS ARM
--------------------------------------
With 400k samples × 1281 features the numpy array is ~2 GB.
sklearn's train_test_split copies the full array, so the naive approach
peaks at 4+ GB and triggers macOS memory-pressure → segfault.

Fix: split by *indices* (3.2 MB) first, then load only the train/val
slices from a memory-mapped .npy file — never holding the full array in RAM.

    y_all:   400k × 8 B  = 3.2 MB     (load fully — trivial)
    X_mmap:  400k × 1281 × 4 B        (memory-mapped from disk)
    X_tr:    320k × 1281 × 4 B = 1.64 GB  (contiguous slice → RAM)
    X_va:    40k  × 1281 × 4 B = 0.21 GB  (contiguous slice → RAM)

Peak RAM during training: X_tr + X_va ≈ 1.85 GB (shared with torch via
from_numpy, no copy).  Test evaluation is skipped — use val accuracy instead.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SEED = 42


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
        import time

        from plasflow2.classify.train import evaluate, save_rf, split_data
        from plasflow2.classify.train import train_rf as _train_rf
        from plasflow2.utils.device import IDX_TO_CLASS

        X = np.load(args.data).astype(np.float32)
        y = np.load(args.labels).astype(np.int64)
        logger.info("Loaded X=%s  y=%s", X.shape, y.shape)

        logger.info("Splitting data …")
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
        logger.info("Train=%d  Val=%d  Test=%d", len(X_tr), len(X_va), len(X_te))

        logger.info("Training Random Forest (500 trees) …")
        t0 = time.time()
        rf = _train_rf(X_tr, y_tr, cv_folds=0)
        logger.info("RF trained in %.1f s", time.time() - t0)

        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]
        result = evaluate(y_te, rf.predict(X_te), class_names=class_names)
        logger.info("Test accuracy: %.4f", result["accuracy"])
        logger.info("\n%s", result["report"])

        save_rf(rf, out_dir / "rf_v2.pkl")

    # ── MLP ──────────────────────────────────────────────────────────────────
    if args.mlp:
        import time

        from sklearn.model_selection import train_test_split  # type: ignore[import]

        from plasflow2.classify.model import save_model
        from plasflow2.classify.train import train_mlp as _train_mlp
        from plasflow2.utils.device import IDX_TO_CLASS

        # ── Step 1: split INDICES (not data) — 3 MB, trivially fast ─────────
        logger.info("Loading labels and splitting by index …")
        y_all = np.load(args.labels).astype(np.int64)
        n = len(y_all)
        logger.info("Total samples: %d", n)

        idx_all = np.arange(n)
        idx_trainval, idx_te, y_trainval, _ = train_test_split(
            idx_all, y_all,
            test_size=0.10,
            stratify=y_all,
            random_state=_SEED,
        )
        idx_tr, idx_va, y_tr, y_va = train_test_split(
            idx_trainval, y_trainval,
            test_size=0.10 / 0.90,
            stratify=y_trainval,
            random_state=_SEED,
        )
        # Free everything we do not need before touching the feature matrix
        del idx_all, y_all, idx_trainval, y_trainval, idx_te
        gc.collect()
        logger.info(
            "Split (index-only): Train=%d  Val=%d  (test skipped — use val accuracy)",
            len(idx_tr), len(idx_va),
        )

        # ── Step 2: load only train+val slices from memory-mapped array ─────
        # X_mmap is disk-backed — reading a slice copies only that slice
        logger.info("Memory-mapping feature file and loading train/val slices …")
        X_mmap = np.load(args.data, mmap_mode="r")
        logger.info("Feature matrix shape: %s  dtype: %s", X_mmap.shape, X_mmap.dtype)

        X_tr = np.ascontiguousarray(X_mmap[idx_tr]).astype(np.float32)
        logger.info("X_tr loaded: %.2f GB", X_tr.nbytes / 1e9)

        X_va = np.ascontiguousarray(X_mmap[idx_va]).astype(np.float32)
        logger.info("X_va loaded: %.2f GB", X_va.nbytes / 1e9)

        # Release the mmap and index arrays before PyTorch allocates anything
        del X_mmap, idx_tr, idx_va
        gc.collect()

        # ── Step 3: train ─────────────────────────────────────────────────────
        logger.info(
            "Training MLP (AdamW + cosine LR + early stopping) …\n"
            "  RAM in use: X_tr=%.2f GB  X_va=%.2f GB  (total ≈%.2f GB)",
            X_tr.nbytes / 1e9, X_va.nbytes / 1e9,
            (X_tr.nbytes + X_va.nbytes) / 1e9,
        )
        t0 = time.time()
        model = _train_mlp(
            X_tr, y_tr, X_va, y_va,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            patience=10,
        )
        elapsed = time.time() - t0
        logger.info("MLP trained in %.1f s (%.1f min)", elapsed, elapsed / 60)

        # ── Step 4: save ──────────────────────────────────────────────────────
        model_path = out_dir / "mlp_v2.pt"
        save_model(model, model_path)
        logger.info("Model saved → %s", model_path)


if __name__ == "__main__":
    main()
