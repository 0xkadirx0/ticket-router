"""Shared paths, data loading, timing and size helpers."""
import json, os, platform, random, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
CLINC_FILE = DATA / "clinc_full.json"
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def download_data():
    DATA.mkdir(exist_ok=True)
    if not CLINC_FILE.exists():
        print("Downloading CLINC150 (data_full.json) ...")
        urllib.request.urlretrieve(CLINC_URL, CLINC_FILE)
    return CLINC_FILE


def load_splits(limit: int | None = None):
    """Return {'train','val','test'} -> (texts, labels). The 'oos' class is included in all
    splits, so every model sees the same 151 labels and is tested on the same 5,500 queries."""
    raw = json.load(open(download_data()))
    splits = {
        "train": raw["train"] + raw["oos_train"],
        "val": raw["val"] + raw["oos_val"],
        "test": raw["test"] + raw["oos_test"],
    }
    out = {}
    rng = random.Random(SEED)
    for k, rows in splits.items():
        rows = list(rows)
        if limit:
            rng.shuffle(rows)
            rows = rows[: max(limit // (1 if k == "train" else 3), 50)]
        out[k] = ([r[0] for r in rows], [r[1] for r in rows])
    return out


def get_labels():
    raw = json.load(open(download_data()))
    intents = sorted({r[1] for r in raw["train"]})
    return intents + ["oos"]  # 150 intents + out-of-scope = 151


def hardware_string():
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    try:
        import torch
        thr = torch.get_num_threads()
    except ImportError:
        thr = "n/a"
    return f"{cpu}; {os.cpu_count()} logical cores; {platform.system()} {platform.release()}; torch threads={thr}; CPU-only inference"


def size_mb(*paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        else:
            total += p.stat().st_size
    return total / (1024 * 1024)


def time_latency(fn, queries, runs: int = 100, warmup: int = 10):
    """Average single-query latency in ms over `runs` timed calls (after warm-up)."""
    for q in queries[:warmup]:
        fn(q)
    ts = []
    for i in range(runs):
        q = queries[i % len(queries)]
        t0 = time.perf_counter(); fn(q); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))
