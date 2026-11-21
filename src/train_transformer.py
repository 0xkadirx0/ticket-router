"""Rung 3 - Transformer. Fine-tunes BERT-base (teacher) with a classification head, or, when
--teacher is given, fine-tunes a smaller student (DistilBERT) by knowledge distillation:
loss = alpha * CE(hard labels) + (1-alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))."""
import argparse, json, time
import torch
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from .common import MODELS, get_labels, load_splits, set_seed
from .textproc import normalize


def encode(tok, texts, max_len):
    return tok([normalize(t) for t in texts], return_tensors="pt", padding=True, truncation=True,
               max_length=max_len, return_token_type_ids=False)


def evaluate(model, tok, texts, y, device, max_len, bs=128):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = {k: v.to(device) for k, v in encode(tok, texts[i:i + bs], max_len).items()}
            correct += (model(**enc).logits.argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(texts)


def main(a):
    set_seed(); device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = get_labels(); l2i = {l: i for i, l in enumerate(labels)}
    s = load_splits(a.limit)
    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = AutoConfig.from_pretrained(a.base, num_labels=len(labels),
                                     id2label=dict(enumerate(labels)), label2id=l2i)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, config=cfg).to(device)
    teacher = None
    if a.teacher:
        teacher = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(device).eval()
    ytr = torch.tensor([l2i[l] for l in s["train"][1]]); yva = torch.tensor([l2i[l] for l in s["val"][1]])
    Xtr = s["train"][0]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(Xtr) + a.bs - 1) // a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); total = 0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            enc = {k: v.to(device) for k, v in encode(tok, [Xtr[j] for j in idx], a.max_len).items()}
            logits = model(**enc).logits; y = ytr[idx].to(device)
            loss = F.cross_entropy(logits, y)
            if teacher is not None:
                with torch.no_grad(): tl = teacher(**enc).logits
                kd = a.T ** 2 * F.kl_div(F.log_softmax(logits / a.T, -1), F.softmax(tl / a.T, -1),
                                         reduction="batchmean")
                loss = a.alpha * loss + (1 - a.alpha) * kd
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += loss.item()
        acc = evaluate(model, tok, s["val"][0], yva, device, a.max_len)
        print(f"epoch {ep}/{a.epochs} | train loss {total/ (len(Xtr)/a.bs):.4f} | val acc {acc:.4f} | {time.time()-t0:.0f}s")
        if acc > best: best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.cpu()
    out = MODELS / a.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    torch.save(model.state_dict(), out / "fp32_state.pt")      # used only to measure fp32 size
    json.dump({"base": a.base, "teacher": a.teacher, "epochs": a.epochs, "lr": a.lr, "batch_size": a.bs,
               "max_len": a.max_len, "alpha": a.alpha if a.teacher else None,
               "T": a.T if a.teacher else None, "best_val_acc": best, "train_seconds": time.time() - t0,
               "device": device}, open(out / "train_info.json", "w"), indent=1)
    print(f"saved {out} | best val acc {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bert-base-uncased"); ap.add_argument("--out", default="bert_base")
    ap.add_argument("--teacher", default=None, help="path of fine-tuned teacher dir -> distillation")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--limit", type=int)
    main(ap.parse_args())
