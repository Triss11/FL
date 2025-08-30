#!/usr/bin/env python3
import argparse, os, shutil, runpy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

TOTAL_SAMPLES = 5888
MAX_USERS = 128

# Always write the remainder here (don’t touch the main CSV)
DEFAULT_TRAIN_TEST_OUT = "/Users/sohinikar/FL/M.Tech_Dissertation/Obfuscated-MalMem2022_train_and_test.csv"

DESTS = {
    ("binary", "fnn"):  Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/FNN_BC_test_data"),
    ("binary", "lstm"): Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/LSTM_BC_test_data"),
    ("multiclass", "fnn"):  Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/FNN_MC_test_data"),
    ("multiclass", "lstm"): Path("/Users/sohinikar/FL/M.Tech_Dissertation/Client/data/LSTM_MC_test_data"),
}
COMMON_LABELS = ["label","Label","class","Class","target","Target","y","Y"]

def norm(s:str)->str: return s.strip().lower()
def even_splits(total:int, parts:int):
    base, rem = total//parts, total%parts
    out=[]; start=0
    for i in range(parts):
        k = base + (1 if i<rem else 0)
        out.append((start, start+k)); start += k
    return out
def to_feature_cols(d:int): return [f"f{i:02d}" for i in range(d)]
def autodetect_label(df:pd.DataFrame)->str:
    for c in COMMON_LABELS:
        if c in df.columns: return c
    raise ValueError(f"Could not auto-detect label column. Tried {COMMON_LABELS}.")
def cast_labels(y:np.ndarray, classification:str)->np.ndarray:
    if classification=="binary":
        uniq = pd.unique(y)
        if len(uniq)!=2: raise ValueError(f"Binary requested but found {len(uniq)} classes: {uniq}")
        mapping={uniq[0]:0.0, uniq[1]:1.0}; print(f"Binary label mapping: {mapping}")
        return pd.Series(y).map(mapping).astype(np.float32).to_numpy()
    uniq = sorted(pd.unique(y), key=lambda v: str(v))
    mapping={v:i for i,v in enumerate(uniq)}; print(f"Multiclass label mapping: {mapping}")
    return pd.Series(y).map(mapping).astype(np.int32).to_numpy()

def load_config(path_hint:str)->tuple[dict,Path]:
    if not os.path.isabs(path_hint):
        cands=[Path(path_hint), Path(__file__).parent/path_hint]
        for c in cands:
            if c.exists(): path_hint=str(c.resolve()); break
    p=Path(path_hint)
    if not p.exists(): raise FileNotFoundError(f"Config not found: {p}")
    return runpy.run_path(str(p)), p

def main():
    ap=argparse.ArgumentParser(description="Create user datasets and a separate train_and_test CSV (remainder).")
    ap.add_argument("--config", default=os.environ.get("SPLIT_CONFIG","/Users/sohinikar/FL/M.Tech_Dissertation/params.py"))
    args=ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    print(f"Using config: {cfg_path}")

    classification = norm(cfg.get("classification","multiclass"))
    model          = norm(cfg.get("model","lstm"))
    users          = int(cfg.get("users",128))
    seed           = int(cfg.get("seed",42))
    dataset_csv    = Path(cfg.get("DATASET_CSV","/Users/sohinikar/FL/M.Tech_Dissertation/Obfuscated-MalMem2022.csv"))
    label_col      = cfg.get("label_column", None)
    clean_existing = bool(cfg.get("clean_existing", True))

    if classification not in ("binary","multiclass"): raise ValueError("classification must be binary|multiclass")
    if model not in ("fnn","lstm"): raise ValueError("model must be fnn|lstm")
    if not (1<=users<=MAX_USERS): raise ValueError(f"users must be in [1,{MAX_USERS}]")
    if not dataset_csv.exists(): raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    # Load full CSV
    df = pd.read_csv(dataset_csv, low_memory=False)
    if label_col is None: label_col = autodetect_label(df)
    if label_col not in df.columns: raise ValueError(f"Label column '{label_col}' not in CSV.")
    print(f"Label column: {label_col}")

    # Sample 5,888 rows (reproducible)
    N_total=len(df)
    if N_total<TOTAL_SAMPLES: raise ValueError(f"Dataset has {N_total} rows; need {TOTAL_SAMPLES}.")
    rng=np.random.default_rng(seed)
    sample_idx=rng.choice(N_total, size=TOTAL_SAMPLES, replace=False)
    mask=np.ones(N_total, dtype=bool); mask[sample_idx]=False

    df_sampled=df.iloc[sample_idx].copy()
    df_remainder=df.loc[mask].copy()  # this is the train+test remainder

    # Audit files
    sampled_csv_path = dataset_csv.with_name(dataset_csv.stem+"_sampled_5888.csv")
    df_sampled.to_csv(sampled_csv_path, index=False)
    print(f"Sampled CSV (exact 5888 removed): {sampled_csv_path}")

    # Always write the remainder to the fixed train_and_test path
    train_test_out = Path(DEFAULT_TRAIN_TEST_OUT)
    train_test_out.parent.mkdir(parents=True, exist_ok=True)
    df_remainder.to_csv(train_test_out, index=False)
    print(f"Remainder CSV (train+test): {train_test_out}  (rows: {len(df_remainder)})")

    # Build numeric feature matrix from sampled rows for per-user splits
    feat_df = df_sampled.drop(columns=[label_col])
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    dropped=[c for c in feat_df.columns if c not in numeric_cols]
    if not numeric_cols: raise ValueError("No numeric feature columns found.")
    if dropped:
        print(f"Dropping non-numeric feature columns ({len(dropped)}): {dropped[:10]}{'...' if len(dropped)>10 else ''}")
    X_sampled = feat_df[numeric_cols].to_numpy(dtype=np.float32)
    y_raw     = df_sampled[label_col].to_numpy()

    # LSTM flatten note (save as (N,D); reshape to (N,1,D) during training)
    if model=="lstm":
        D=X_sampled.shape[1]
        X_to_save=X_sampled.reshape((X_sampled.shape[0],1,D))[:,0,:]
    else:
        X_to_save=X_sampled

    # Cast labels to requested type
    y_to_save = cast_labels(y_raw, classification)

    # Write per-user folders
    dest_base = DESTS[(classification, model)]
    dest_base.mkdir(parents=True, exist_ok=True)
    if clean_existing:
        for p in dest_base.glob("user_*"):
            if p.is_dir(): shutil.rmtree(p)

    slices = even_splits(TOTAL_SAMPLES, users)
    cols = to_feature_cols(X_to_save.shape[1])

    for i,(a,b) in enumerate(slices, start=1):
        udir = dest_base / f"user_{i:03d}"
        udir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_to_save[a:b], columns=cols).to_csv(udir/"X.csv", index=False)
        pd.DataFrame({"label": y_to_save[a:b]}).to_csv(udir/"y.csv", index=False)

    counts=[b-a for (a,b) in slices]
    print("✅ Done.")
    print(f"   Users: {users} → base: {dest_base}")
    print(f"   Numeric features kept: {len(numeric_cols)}")
    print(f"   Per-user sizes (min/median/max): {min(counts)}/{int(np.median(counts))}/{max(counts)}")
    print(f"   Sampled X shape: ({TOTAL_SAMPLES}, {X_to_save.shape[1]}), y shape: ({TOTAL_SAMPLES},)")
    if model=="lstm":
        print("   Note: reshape X to (N, 1, D) at training time.")

if __name__=="__main__":
    main()
