#!/usr/bin/env python3
import os, sys, time, json, shutil, runpy, argparse, shlex, subprocess
from pathlib import Path
from typing import Optional, List

# ---- Fixed paths you gave ----
PARAMS_PY  = Path("/Users/sohinikar/FL/M.Tech_Dissertation/params.py")
MAKE_USERS = Path("/Users/sohinikar/FL/M.Tech_Dissertation/make_users.py")
NOTEBOOK   = Path("/Users/sohinikar/FL/M.Tech_Dissertation/Server/Binary_malware_detection_centralized_ CIC-MalMem-2022.ipynb")

FNN_OUT    = Path("/Users/sohinikar/FL/M.Tech_Dissertation/Server/global_FNN_model.keras")
LSTM_OUT   = Path("/Users/sohinikar/FL/M.Tech_Dissertation/Server/global_lstm_model.keras")
SERVER_DIR = Path("/Users/sohinikar/FL/M.Tech_Dissertation/Server")
REPO_ROOT  = Path(os.environ.get("REPO_ROOT", "/Users/sohinikar/FL"))  # git root

SEARCH_DIRS = [SERVER_DIR]
LOG_DIR     = Path("/Users/sohinikar/FL/M.Tech_Dissertation/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH    = LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

def stream_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """Run a command and stream stdout/stderr live to console + log."""
    print(f"➤ exec: {' '.join(shlex.quote(c) for c in cmd)}")
    with open(LOG_PATH, "a", buffering=1) as logf:
        logf.write(f"\n=== RUN: {' '.join(cmd)} (cwd={cwd}) ===\n")
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")

def run_cmd_capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr}")
    return (cp.stdout or "").strip()

def load_params(py_path: Path) -> dict:
    return runpy.run_path(str(py_path))

def newest_keras(base_dirs: List[Path], hint: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for base in base_dirs:
        if not base.exists(): continue
        for f in base.rglob("*.keras"):
            if hint and hint not in f.name.lower(): continue
            candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def copy_if_exists(src: Optional[Path], dst: Path, label: str) -> bool:
    if src is None:
        print(f"⚠️  Could not find a candidate .keras file for {label}.")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"✅ {label}: {src.name} → {dst}")
    return True

def have(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except Exception:
        return False

def validate_paths_or_die():
    problems = []
    if not PARAMS_PY.exists():  problems.append(f"Missing params.py: {PARAMS_PY}")
    if not MAKE_USERS.exists(): problems.append(f"Missing make_users.py: {MAKE_USERS}")
    if not NOTEBOOK.exists():   problems.append(f"Missing notebook: {NOTEBOOK}")
    if not REPO_ROOT.exists():  problems.append(f"Missing repo root: {REPO_ROOT}")
    if problems:
        print("\n❌ Path validation failed:")
        for p in problems: print("   -", p)
        sys.exit(1)

def step1_make_users():
    print("\n=== Step 1: Make user datasets (5,888 split) ===")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SPLIT_CONFIG"] = str(PARAMS_PY)
    stream_cmd([sys.executable, str(MAKE_USERS), "--config", str(PARAMS_PY)], env=env)
    print("✅ Step 1 complete.")

def step2_run_notebook():
    print("\n=== Step 2: Execute global model notebook ===")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Prefer papermill; else nbconvert
    if have([sys.executable, "-m", "papermill", "--version"]):
        stream_cmd([sys.executable, "-m", "papermill", str(NOTEBOOK), str(NOTEBOOK), "-k", "python3"], env=env)
    elif have(["jupyter", "--version"]):
        stream_cmd(["jupyter", "nbconvert", "--to", "notebook", "--inplace", "--execute", str(NOTEBOOK)], env=env)
    else:
        raise RuntimeError(
            "Neither papermill nor jupyter is installed.\n"
            f"Install one of:\n  {sys.executable} -m pip install papermill\n"
            f"or\n  {sys.executable} -m pip install jupyter nbconvert\n"
        )
    print("✅ Notebook executed.")

def step3_sync_models():
    print("\n=== Step 3: Ensure models at required paths ===")
    ok_fnn = FNN_OUT.exists()
    ok_lstm = LSTM_OUT.exists()
    if not ok_fnn:
        cand = newest_keras(SEARCH_DIRS, hint="fnn") or newest_keras(SEARCH_DIRS)
        ok_fnn = copy_if_exists(cand, FNN_OUT, "FNN model")
    else:
        print(f"✔ Found FNN: {FNN_OUT}")
    if not ok_lstm:
        cand = newest_keras(SEARCH_DIRS, hint="lstm") or newest_keras(SEARCH_DIRS)
        ok_lstm = copy_if_exists(cand, LSTM_OUT, "LSTM model")
    else:
        print(f"✔ Found LSTM: {LSTM_OUT}")
    if not (ok_fnn or ok_lstm):
        raise RuntimeError("No .keras artifacts found to sync.")
    print("✅ Model sync complete.")

def step4_git_commit_push(message: str):
    print("\n=== Step 4: Git add / commit / push ===")
    # Basic checks
    run_cmd_capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO_ROOT)
    branch = run_cmd_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT)
    print(f"Repo: {REPO_ROOT}  Branch: {branch}")
    # Add
    stream_cmd(["git", "add", "-A"], cwd=REPO_ROOT)
    # Commit if staged changes exist
    staged = run_cmd_capture(["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT)
    if staged.strip():
        stream_cmd(["git", "commit", "-m", message], cwd=REPO_ROOT)
    else:
        print("ℹ️  No staged changes; skipping commit.")
    # Push, auto rebase if needed
    try:
        stream_cmd(["git", "push", "origin", branch], cwd=REPO_ROOT)
    except RuntimeError:
        print("↩️  Push rejected. Running 'git pull --rebase' and retrying…")
        stream_cmd(["git", "pull", "--rebase", "origin", branch], cwd=REPO_ROOT)
        stream_cmd(["git", "push", "origin", branch], cwd=REPO_ROOT)
    print("✅ Git push complete.")

def main():
    ap = argparse.ArgumentParser(description="Verbose pipeline: make users → run notebook → sync models → git push")
    ap.add_argument("--commit-message", default=None, help="Commit message for git.")
    args = ap.parse_args()

    print(f"Python: {sys.executable}")
    print(f"CWD   : {os.getcwd()}")
    print(f"Log   : {LOG_PATH}")
    print("Validating paths…")
    validate_paths_or_die()

    # Show a quick snapshot of params
    cfg = load_params(PARAMS_PY)
    print("Config snapshot:", json.dumps({
        "classification": cfg.get("classification"),
        "model": cfg.get("model"),
        "users": cfg.get("users"),
        "dataset": str(cfg.get("DATASET_CSV", "")),
    }, indent=2))

    t0 = time.time()
    step1_make_users()
    step2_run_notebook()
    step3_sync_models()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = args.commit_message or f"pipeline: datasets + global models | cfg={cfg.get('classification')},{cfg.get('model')},{cfg.get('users')} | {ts}"
    step4_git_commit_push(msg)
    print(f"\n🎉 Done in {time.time()-t0:.1f}s")
    print(f"Artifacts:\n  - {FNN_OUT} ({'OK' if FNN_OUT.exists() else 'MISSING'})\n  - {LSTM_OUT} ({'OK' if LSTM_OUT.exists() else 'MISSING'})")
    print(f"Full log: {LOG_PATH}")

if __name__ == "__main__":
    main()
