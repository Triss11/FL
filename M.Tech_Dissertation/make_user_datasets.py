#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# ---- import your inputs ----
import params  # reads: classification, model, users, seed, DATA_DIR

TOTAL_SAMPLES = 5888  # fixed per your requirement
MAX_USERS = 128

# ---- destination roots (by classification + model) ----
DESTS = {
    ("binary", "fnn"):  Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/FNN_BC_test_data"),
    ("binary", "lstm"): Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/LSTM_BC_test_data"),
    ("multiclass", "fnn"):  Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/FNN_MC_test_data"),
    ("multiclass", "lstm"): Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/LSTM_MC_test_data"),
}

# If True, remove existing user_* folders in the target before writing new ones
CLEAN_EXISTING = True


# ---------- Utilities ----------
def norm(s: str) -> str:
    return s.strip().lower()

def load_dataset(data_dir: Path):
    """
    Try loading X_test, y_test from:
      1) .npy files: X_test.npy, y_test.npy
      2) .csv files: X_test.csv, y_test.csv
         - X_test.csv: numeric columns (features)
         - y_test.csv: either single column or has a 'label' column
    Returns: (X: np.ndarray [N, D], y: np.ndarray [N])
    """
    x_npy = data_dir / "X_test.npy"
    y_npy = data_dir / "y_test.npy"
    x_csv = data_dir / "X_test.csv"
    y_csv = data_dir / "y_test.csv"

    if x_npy.exists() and y_npy.exists():
        X = np.load(x_npy)
        y = np.load(y_npy)
        y = y.reshape(-1)
        return np.asarray(X), np.asarray(y)

    if x_csv.exists() and y_csv.exists():
        X_df = pd.read_csv(x_csv)
        y_df = pd.read_csv(y_csv)
        # Allow either a single column or a named column 'label'
        if "label" in y_df.columns:
            y = y_df["label"].to_numpy()
        else:
            # take first column if only one exists
            if len(y_df.columns) != 1:
                raise ValueError(f"{y_csv} must have a single column or a 'label' column.")
            y = y_df.iloc[:, 0].to_numpy()
        return X_df.to_numpy(), y.reshape(-1)

    raise FileNotFoundError(
        f"Could not find dataset in {data_dir}. "
        f"Expected either X_test.npy+y_test.npy or X_test.csv+y_test.csv."
    )

def sample_indices(n: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)

def even_splits(total: int, parts: int):
    """
    Returns a list of slice (start, end) pairs that split `total` into `parts`
    as evenly as possible. First `remainder` parts get +1.
    """
    base = total // parts
    rem = total % parts
    starts = []
    s = 0
    for i in range(parts):
        add = base + (1 if i < rem else 0)
        starts.append((s, s + add))
        s += add
    return starts

def to_feature_cols(d: int, prefix: str = "f"):
    return [f"{prefix}{i:02d}" for i in range(d)]


# ---------- Main ----------
def main():
    classification = norm(params.classification)
    model = norm(params.model)
    users = int(params.users)
    seed = int(getattr(params, "seed", 42))
    data_dir = Path(getattr(params, "DATA_DIR", "."))

    if classification not in ("binary", "multiclass"):
        raise ValueError("classification must be 'binary' or 'multiclass'")
    if model not in ("fnn", "lstm"):
        raise ValueError("model must be 'fnn' or 'lstm'")
    if not (1 <= users <= MAX_USERS):
        raise ValueError(f"users must be in [1, {MAX_USERS}]")

    dest_base = DESTS[(classification, model)]

    # 1) Load full dataset
    X_full, y_full = load_dataset(data_dir)
    N, D = X_full.shape
    if len(y_full) != N:
        raise ValueError(f"X_test rows ({N}) != y_test rows ({len(y_full)})")
    if N < TOTAL_SAMPLES:
        raise ValueError(f"Dataset has only {N} rows; need at least {TOTAL_SAMPLES}.")

    # 2) Randomly sample 5888 without replacement
    idx = sample_indices(N, TOTAL_SAMPLES, seed=seed)
    X_sampled = X_full[idx]
    y_sampled = y_full[idx]

    # 3) If LSTM: reshape to (N, 1, D) per your spec (we’ll save flattened to CSV)
    if model == "lstm":
        X_lstm = X_sampled.reshape((X_sampled.shape[0], 1, X_sampled.shape[1])).astype(np.float32)
        # For CSV saving: flatten time dimension (1) -> (N, D)
        X_to_save = X_lstm[:, 0, :]
    else:
        X_to_save = X_sampled.astype(np.float32)

    # 4) Sanitize labels:
    if classification == "binary":
        # cast to float 0/1
        if y_sampled.dtype.kind in {"U", "S", "O"}:  # strings -> indices -> 0/1 floats
            _, inv = np.unique(y_sampled, return_inverse=True)
            y_to_save = inv.astype(np.float32)
        else:
            y_to_save = y_sampled.astype(np.float32)
    else:
        # multiclass -> integers 0..C-1
        if y_sampled.dtype.kind in {"U", "S", "O"}:
            _, inv = np.unique(y_sampled, return_inverse=True)
            y_to_save = inv.astype(np.int32)
        elif y_sampled.dtype.kind == "f":
            y_to_save = np.rint(y_sampled).astype(np.int32)
        else:
            y_to_save = y_sampled.astype(np.int32)

    # 5) Prepare destination; optional clean
    dest_base.mkdir(parents=True, exist_ok=True)
    if CLEAN_EXISTING:
        for p in dest_base.glob("user_*"):
            if p.is_dir():
                shutil.rmtree(p)

    # 6) Split into user chunks and write CSVs
    slices = even_splits(TOTAL_SAMPLES, users)
    cols = to_feature_cols(D)

    for i, (a, b) in enumerate(slices, start=1):
        user_dir = dest_base / f"user_{i:03d}"
        user_dir.mkdir(parents=True, exist_ok=True)

        # Features
        pd.DataFrame(X_to_save[a:b], columns=cols).to_csv(user_dir / "X.csv", index=False)
        # Labels
        pd.DataFrame({"label": y_to_save[a:b]}).to_csv(user_dir / "y.csv", index=False)

    # 7) Final print
    per_user_counts = [b - a for (a, b) in slices]
    print(f"✅ Wrote {users} user folders under: {dest_base}")
    print(f"   Samples per user (min/median/max): {min(per_user_counts)}/{int(np.median(per_user_counts))}/{max(per_user_counts)}")
    print(f"   Total written: {sum(per_user_counts)} (expected {TOTAL_SAMPLES})")
    print(f"   Shape: X -> ({TOTAL_SAMPLES}, {D}), y -> ({TOTAL_SAMPLES},)")
    if model == "lstm":
        print("   Note: Saved X.csv flattened from (N, 1, D); reshape back to (N, 1, D) during training.")

if __name__ == "__main__":
    main()
