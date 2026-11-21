"""One command to reproduce everything: python run_all.py   (add --quick for a tiny smoke test)
Order: data -> baseline -> word2vec+FFN -> BERT-base teacher -> DistilBERT (KD) -> INT8 -> evaluate -> slides.
BERT fine-tuning is slow on CPU: use a GPU (e.g. free Colab T4) for steps 3-4."""
import subprocess, sys
quick = "--quick" in sys.argv
lim = ["--limit", "600"] if quick else []
ep = ["--epochs", "1"] if quick else []
py = sys.executable
steps = [
    [py, "-m", "src.train_baseline", *lim],
    [py, "-m", "src.train_ffn", *lim, *(["--epochs", "5"] if quick else [])],
    [py, "-m", "src.train_transformer", "--base", "bert-base-uncased", "--out", "bert_base", *ep, *lim],
    [py, "-m", "src.train_transformer", "--base", "distilbert-base-uncased", "--out", "distilbert_kd",
     "--teacher", "models/bert_base", "--alpha", "0.5", "--T", "2", *ep, *lim],
    [py, "-m", "src.quantize"],
    [py, "-m", "src.evaluate", *lim],
    [py, "make_report.py"],
]
for s in steps:
    print("\n>>>", " ".join(s)); subprocess.run(s, check=True)
print("\nDone. Run the app with: python app.py")
