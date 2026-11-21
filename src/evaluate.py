"""Evaluate every model on the SAME held-out test set (CLINC150 test + out-of-scope test = 5,500
queries). Writes results/results.json (the only source of numbers for the slides)."""
import argparse, json, random, time, datetime
import numpy as np
from .common import MODELS, RESULTS, hardware_string, load_splits, set_seed, size_mb, time_latency
from .predictors import BaselinePredictor, FFNPredictor, TransformerPredictor
from .textproc import build_lexicon, correct_text, normalize


def metrics(pred, gold):
    pred, gold = np.array(pred), np.array(gold)
    oos = gold == "oos"
    return {"accuracy": float((pred == gold).mean()),
            "in_scope_accuracy": float((pred[~oos] == gold[~oos]).mean()),
            "oos_recall": float((pred[oos] == "oos").mean())}


def inject_typo(text, rng):
    words = text.split(); cands = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not cands: return text
    i = rng.choice(cands); w = words[i]; k = rng.randrange(1, len(w) - 1); op = rng.choice("dsr")
    if op == "d": w = w[:k] + w[k + 1:]
    elif op == "s": w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    else: w = w[:k] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[k + 1:]
    words[i] = w; return " ".join(words)


def load_models():
    M = []
    def add(name, rung, build, files):
        try: M.append((name, rung, build(), files))
        except Exception as e: print(f"[skip] {name}: {e}")
    add("tf-idf + LogReg", "1 classical baseline", BaselinePredictor, lambda m: [m.path])
    add("word2vec + FFN (SGD)", "2 neural network", FFNPredictor, lambda m: m.files())
    add("BERT-base fine-tuned (fp32)", "3 Transformer",
        lambda: TransformerPredictor(MODELS / "bert_base", "BERT-base"), lambda m: [m.weight_file()])
    add("DistilBERT-KD (fp32)", "3b distilled student",
        lambda: TransformerPredictor(MODELS / "distilbert_kd", "DistilBERT-KD"), lambda m: [m.weight_file()])
    add("DistilBERT-KD (INT8)", "4 quantized Transformer",
        lambda: TransformerPredictor(MODELS / "distilbert_kd_int8", "DistilBERT-KD-int8", int8=True),
        lambda m: [m.weight_file()])
    return M


def main(limit=None, runs=100, typo_n=1000):
    set_seed()
    try:
        import torch; torch.set_num_threads(1)
    except ImportError: pass
    s = load_splits(limit); X, y = s["test"]
    rows, errors = [], {}
    models = load_models()
    for name, rung, m, files in models:
        t0 = time.time(); P = m.predict_proba(X)
        pred = [m.labels[i] for i in P.argmax(1)]
        r = {"model": name, "rung": rung, **metrics(pred, y), "size_mb": size_mb(*files(m))}
        r["latency_ms"], r["latency_std_ms"] = time_latency(lambda q: m.predict_proba([q]), X, runs)
        r["latency_runs"] = runs
        rows.append(r); print(f"{name}: acc {r['accuracy']:.4f} oos-recall {r['oos_recall']:.4f} "
                              f"{r['size_mb']:.1f}MB {r['latency_ms']:.1f}ms ({time.time()-t0:.0f}s)")
        errors[name] = [{"text": X[i], "gold": y[i], "pred": pred[i], "conf": float(P[i].max())}
                        for i in range(len(X)) if pred[i] != y[i]]
    final = next((m for n, r_, m, f in models if r_.startswith("4")), None)
    typo = None
    if final is not None:                       # measured benefit of edit-distance correction
        rng = random.Random(7); lex = build_lexicon(load_splits(limit)["train"][0])
        idx = rng.sample(range(len(X)), min(typo_n, len(X)))
        noisy = [inject_typo(normalize(X[i]), rng) for i in idx]
        fixed = [correct_text(t, lex)[0] for t in noisy]
        g = [y[i] for i in idx]
        acc = lambda T: float(np.mean([final.labels[i] == c for i, c in zip(final.predict_proba(T).argmax(1), g)]))
        typo = {"n": len(idx), "clean": acc([normalize(X[i]) for i in idx]), "typos": acc(noisy),
                "typos_corrected": acc(fixed), "model": final.name}
        print("typo experiment:", typo)
    RESULTS.mkdir(exist_ok=True)
    json.dump({"date": datetime.date.today().isoformat(), "hardware": hardware_string(),
               "n_test": len(y), "n_test_oos": int(sum(c == "oos" for c in y)), "rows": rows, "typo": typo,
               "errors": {k: v[:60] for k, v in errors.items()}},
              open(RESULTS / "results.json", "w"), indent=1)
    with open(RESULTS / "results.md", "w") as f:
        f.write("| Model | Accuracy | OOS recall | Size (MB) | Latency (ms) |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['accuracy']*100:.1f}% | {r['oos_recall']*100:.1f}% | "
                    f"{r['size_mb']:.1f} | {r['latency_ms']:.1f} ± {r['latency_std_ms']:.1f} |\n")
    print("wrote results/results.json and results/results.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int)
    ap.add_argument("--runs", type=int, default=100); ap.add_argument("--typo-n", type=int, default=1000)
    a = ap.parse_args(); main(a.limit, a.runs, a.typo_n)
