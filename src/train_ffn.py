"""Rung 2 - neural network trained by us: word2vec (trained on the training messages) -> averaged
embedding -> feedforward network, trained with mini-batch SGD + backpropagation (PyTorch autograd)."""
import argparse, json, time
import numpy as np, torch
from torch import nn
from gensim.models import Word2Vec
from .common import MODELS, get_labels, load_splits, set_seed
from .textproc import normalize


def make_ffn(dim, hidden, n_out, p=0.3):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_out))


def avg_embed(kv, texts):
    out = np.zeros((len(texts), kv.vector_size), dtype="float32")
    for i, t in enumerate(texts):
        vs = [kv[w] for w in normalize(t).split() if w in kv]
        if vs: out[i] = np.mean(vs, axis=0)
    return out


def main(dim=100, hidden=256, epochs=80, lr=0.1, batch=64, limit=None):
    set_seed(); out = MODELS / "ffn"; out.mkdir(parents=True, exist_ok=True)
    s = load_splits(limit); labels = get_labels(); l2i = {l: i for i, l in enumerate(labels)}
    sents = [normalize(t).split() for t in s["train"][0]]
    w2v = Word2Vec(sents, vector_size=dim, window=5, min_count=1, sg=1, epochs=40, workers=1, seed=42)
    kv = w2v.wv; kv.save(str(out / "w2v.kv"))
    X = torch.tensor(avg_embed(kv, s["train"][0])); y = torch.tensor([l2i[l] for l in s["train"][1]])
    Xv = torch.tensor(avg_embed(kv, s["val"][0])); yv = torch.tensor([l2i[l] for l in s["val"][1]])
    net = make_ffn(dim, hidden, len(labels))
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)   # SGD + backprop
    lossf = nn.CrossEntropyLoss(); best, best_state = 0, None; t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); lossf(net(X[idx]), y[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): acc = (net(Xv).argmax(1) == yv).float().mean().item()
        if acc > best: best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 10 == 0: print(f"epoch {ep:3d} | val acc {acc:.4f} | best {best:.4f}")
    torch.save({"state": best_state, "dim": dim, "hidden": hidden}, out / "ffn.pt")
    json.dump(labels, open(out / "labels.json", "w"))
    print(f"FFN trained in {time.time()-t0:.0f}s | best val acc {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); main(epochs=a.epochs, limit=a.limit)
