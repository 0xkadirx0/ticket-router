"""Builds the 12-slide PDF (presentation.pdf) from results/results.json + config.json.
Every number on the slides is read from results.json (your own run). With no results yet, the
metrics show '-' and a DRAFT watermark is printed, so no number can be invented."""
import json, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent
W, H = 960, 540
NAVY, ACCENT, GREY = colors.HexColor("#1f3a5f"), colors.HexColor("#c62828"), colors.HexColor("#555555")
fd = Path(font_manager.findfont("DejaVu Sans")).parent
pdfmetrics.registerFont(TTFont("DV", str(fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(fd / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(fd / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

cfg = json.load(open(ROOT / "config.json"))
RES = ROOT / "results" / "results.json"
res = json.load(open(RES)) if RES.exists() else None
DRAFT = res is None
rows = res["rows"] if res else []
def row(prefix): return next((r for r in rows if r["rung"].startswith(prefix)), None)
base, ffn, bert, dist, q8 = row("1"), row("2"), row("3 "), row("3b"), row("4")
pct = lambda v: "-" if v is None else f"{v*100:.1f}%"
num = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
g = lambda r, k: None if r is None else r[k]
def info(d):
    p = ROOT / "models" / d / "train_info.json"
    return json.load(open(p)) if p.exists() else {}
ibert, idist = info("bert_base"), info("distilbert_kd")
unfilled = [k for k, v in cfg.items() if "FILL" in str(v)]

def sty(size=15, bold=False, color=colors.black, lead=None):
    return ParagraphStyle("s", fontName="DVB" if bold else "DV", fontSize=size, textColor=color,
                          leading=lead or size * 1.3)
def para(c, text, x, y_top, w, size=15, bold=False, color=colors.black):
    p = Paragraph(text, sty(size, bold, color)); _, h = p.wrap(w, 1000); p.drawOn(c, x, y_top - h); return y_top - h
def bullets(c, items, x=50, y=455, w=860, size=16, gap=9):
    for it in items:
        y = para(c, "•&nbsp;&nbsp;" + it, x, y, w, size) - gap
    return y
def slide(c, n, section, title):
    c.setFillColor(NAVY); c.rect(0, H - 70, W, 70, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("DVB", 24); c.drawString(40, H - 45, title)
    c.setFont("DV", 11); c.drawRightString(W - 30, H - 25, section)
    c.setFillColor(GREY); c.setFont("DV", 9)
    c.drawString(40, 18, "SWE015 · Support Ticket Router"); c.drawRightString(W - 30, 18, str(n))
    if DRAFT:
        c.saveState(); c.setFillColor(colors.Color(0.8, 0, 0, alpha=0.12)); c.setFont("DVB", 90)
        c.translate(W / 2, H / 2 - 20); c.rotate(25); c.drawCentredString(0, 0, "DRAFT"); c.restoreState()
def table(c, data, x, y_top, widths, size=11, header=NAVY, hl_row=None):
    cell = ParagraphStyle("c", fontName="DV", fontSize=size, leading=size * 1.25)
    hdr = ParagraphStyle("h", fontName="DVB", fontSize=size, leading=size * 1.25, textColor=colors.white)
    d = [[Paragraph(str(v), hdr if i == 0 else cell) for v in r] for i, r in enumerate(data)]
    t = Table(d, colWidths=widths)
    st = [("BACKGROUND", (0, 0), (-1, 0), header), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")])]
    if hl_row: st.append(("BACKGROUND", (0, hl_row), (-1, hl_row), colors.HexColor("#ffe0e0")))
    t.setStyle(TableStyle(st)); _, h = t.wrap(sum(widths), 1000); t.drawOn(c, x, y_top - h); return y_top - h

# trade-off chart ------------------------------------------------------------------------------
def chart():
    if not rows: return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r in rows:
        ax.scatter(r["latency_ms"], r["accuracy"] * 100, s=40 + r["size_mb"] * 2.2, alpha=.65,
                   color="#c62828" if r["rung"].startswith("4") else "#1f3a5f")
        ax.annotate(r["model"].replace(" (", "\n("), (r["latency_ms"], r["accuracy"] * 100), fontsize=7,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("latency per query (ms, CPU)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs latency (bubble area = size on disk)", fontsize=10); ax.grid(alpha=.3)
    p = ROOT / "results" / "tradeoff.png"; fig.savefig(p, dpi=170, bbox_inches="tight"); return p

c = canvas.Canvas(str(ROOT / "presentation.pdf"), pagesize=(W, H))
c.setTitle("SWE015 Homework - Customer Support Ticket Router")

# 1 title
c.setFillColor(NAVY); c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(colors.white); c.setFont("DVB", 36); c.drawString(50, 440, "Customer Support Ticket Router")
c.setFont("DV", 18); c.drawString(50, 408, "SWE015 · Building an Intelligent Application · Use Case 2")
y = 340
for lab, val in [("Name", cfg["student_name"]), ("Student ID", cfg["student_id"]),
                 ("Use case", "2 - Customer Support Ticket Router"), ("Repository", cfg["repo_url"]),
                 ("Demo video", cfg["video_url"])]:
    c.setFont("DVB", 15); c.drawString(50, y, lab + ":"); c.setFont("DV", 15); c.drawString(190, y, str(val)); y -= 38
if DRAFT:
    c.setFillColor(colors.HexColor("#ffb3b3")); c.setFont("DVB", 14); c.drawString(50, 60, "DRAFT - run `python run_all.py` to fill in measured results")
c.showPage()

# 2 introduction
slide(c, 2, "INTRODUCTION", "Problem, users and success criteria")
bullets(c, [
    "<b>Problem:</b> a home-appliance brand receives thousands of support messages per day; each must reach the right team instantly, and unclear or off-topic messages must not be mis-routed.",
    "<b>Who it is for:</b> the support operation (routing without manual triage) and customers (faster first response).",
    "<b>Use case:</b> Use Case 2 - free-text message → normalize → intent + confidence → team. Low-confidence / out-of-scope → <b>human agent</b>; greetings → rule-based reply.",
    "<b>Success =</b> (1) a working app that routes new messages live; (2) a four-model ladder compared on the same test set; (3) the quantized Transformer clearly smaller and faster than the fp32 Transformer at a small accuracy cost; (4) honest measurement of all of it.",
    "<b>Data caveat:</b> CLINC150 is general-assistant data used as a public proxy corpus; its 10 domains are mapped to support teams (src/teams.py).",
], size=15)
c.showPage()

# 3 data
slide(c, 3, "DEVELOPMENT", "Data: CLINC150 (plus out-of-scope)")
n_oos = res["n_test_oos"] if res else None
table(c, [["Split", "Queries", "Content"],
          ["Train", "15,100", "15,000 in-scope (150 intents × 100) + 100 out-of-scope"],
          ["Validation", "3,100", "3,000 in-scope (150 × 20) + 100 out-of-scope - model selection only"],
          ["Test", "5,500", "4,500 in-scope (150 × 30) + 1,000 out-of-scope - identical for all models"]],
      50, 450, [120, 100, 640], 13)
bullets(c, [
    "<b>Source:</b> Larson et al. (2019), CLINC150 'data_full' release, github.com/clinc/oos-eval (downloaded by src/common.py).",
    "<b>Labels:</b> 150 intents + 1 'oos' class = 151 outputs for every model.",
    "<b>Pre-processing</b> (src/textproc.py): URL removal, model-code masking (e.g. DW-6534), lowercasing, repeated-letter squeeze, punctuation cleanup - applied identically in training and testing.",
    "<b>Not measured on Turkish:</b> the corpus is English; the pipeline is language-agnostic but Turkish data would be needed to evaluate it.",
], y=330, size=14)
c.showPage()

# 4 architecture
slide(c, 4, "DEVELOPMENT", "Architecture (matches the code)")
img = ROOT / "docs" / "architecture.png"
if img.exists(): c.drawImage(str(img), 30, 40, width=900, height=900 * 5.6 / 13 * 1.0, preserveAspectRatio=True, mask="auto")
c.showPage()

# 5 technique map
slide(c, 5, "DEVELOPMENT", "Technique map")
table(c, [["Component", "Technique", "Area", "File"],
          ["Message normalization, model-code extraction", "Regular expressions, text normalization", "B", "src/textproc.py"],
          ["Product-name typo correction", "Minimum edit distance (DP) vs. lexicon", "B", "src/textproc.py"],
          ["Greetings / small talk", "ELIZA-style regex rules with capture groups", "B", "src/textproc.py"],
          ["Baseline classifier", "tf-idf + logistic regression", "B", "src/train_baseline.py"],
          ["Word embeddings", "word2vec (skip-gram) trained on the training messages", "B", "src/train_ffn.py"],
          ["Neural network", "FFN, mini-batch SGD + backpropagation, dropout", "A", "src/train_ffn.py"],
          ["Transformer", "BERT-base fine-tuned with a task head; WordPiece tokenizer", "C", "src/train_transformer.py"],
          ["Smaller student", "Knowledge distillation (BERT → DistilBERT)", "D", "src/train_transformer.py"],
          ["Quantized model", "Dynamic INT8 quantization of nn.Linear layers", "D", "src/quantize.py"],
          ["Benchmark", "Size, latency (100 runs), accuracy, OOS recall", "D", "src/evaluate.py"]],
      40, 460, [250, 400, 50, 220], 13)
c.showPage()

# 6 models
slide(c, 6, "DEVELOPMENT", "The four models and how each was trained")
def hp(i, keys):
    return ", ".join(f"{k} {i[k]}" for k in keys if i.get(k) is not None) or "see models/*/train_info.json"
table(c, [["Rung", "Model", "Training"],
          ["1 Classical", "tf-idf (1-2-grams) + logistic regression (C=20)", "scikit-learn, fit on 15,100 training queries"],
          ["2 Neural net", "word2vec (100-d) averaged → FFN 100-256-151", "word2vec 40 epochs; FFN: SGD lr 0.1, momentum 0.9, batch 64, 80 epochs, dropout 0.3; best-val checkpoint"],
          ["3 Transformer", "BERT-base-uncased + classification head (110 M params)", hp(ibert, ["epochs", "lr", "batch_size", "max_len"]) + ", AdamW, warm-up + linear decay; best-val checkpoint"],
          ["3b Student", "DistilBERT-base-uncased distilled from the BERT model", hp(idist, ["alpha", "T", "epochs", "lr"]) + "; loss = α·CE + (1-α)·T²·KL"],
          ["4 Quantized", "DistilBERT student, dynamic INT8 on all nn.Linear", "post-training, no retraining; weights int8, activations quantized on the fly"]],
      40, 455, [110, 330, 450], 13)
para(c, "All models are selected on the validation split and evaluated once on the shared test split.", 40, 130, 880, 13, color=GREY)
c.showPage()

# 7 quantization
slide(c, 7, "DEVELOPMENT", "Quantization method (Area D)")
fp_s, q_s = g(dist, "size_mb"), g(q8, "size_mb")
bullets(c, [
    "<b>Method:</b> post-training <i>dynamic</i> quantization (torch.quantization.quantize_dynamic) of every nn.Linear layer to INT8; activations are quantized on the fly at inference.",
    "<b>Why dynamic:</b> Transformers are limited by compute and memory bandwidth of their large weight matrices; it needs one line of code, no calibration data and no retraining.",
    "<b>Affine map:</b> f = S·(q − Z), with scale S and zero point Z, clamped to the INT8 range.",
    "<b>Distillation first:</b> DistilBERT (6 layers) learns from the BERT-base teacher's softened outputs, then is quantized - the same two-step recipe as the course lecture.",
    f"<b>Measured effect (this run):</b> student fp32 → INT8: size {num(fp_s)} → {num(q_s)} MB; latency {num(g(dist,'latency_ms'))} → {num(g(q8,'latency_ms'))} ms; accuracy {pct(g(dist,'accuracy'))} → {pct(g(q8,'accuracy'))}.",
], size=15)
c.showPage()

# 8 results
slide(c, 8, "DEVELOPMENT", "Results table (same 5,500-query test set)")
data = [["Model", "Accuracy", "OOS recall", "Size (MB)", "Latency (ms) mean ± std"]]
for r in ([base, ffn, bert, dist, q8] if rows else [None]):
    data.append(["-"] * 5 if r is None else [r["model"], pct(r["accuracy"]), pct(r["oos_recall"]), num(r["size_mb"]),
                                              f"{r['latency_ms']:.1f} ± {r['latency_std_ms']:.1f}"])
yb = table(c, data, 40, 455, [330, 110, 110, 110, 220], 13, hl_row=(len(data) - 1 if rows else None))
hw = res["hardware"] if res else "-"
para(c, f"Latency: single query, mean of {rows[0]['latency_runs'] if rows else '≥100'} runs after 10 warm-up runs. Hardware: {hw}. "
        "OOS recall = share of the 1,000 out-of-scope test queries predicted as 'oos'. Size = serialized weights on disk.", 40, yb - 15, 880, 11, color=GREY)
c.showPage()

# 9 comparison
slide(c, 9, "CONCLUSION", "What the comparison showed")
p = chart()
if p: c.drawImage(str(p), 30, 70, width=450, height=330, preserveAspectRatio=True, mask="auto")
facts = []
if rows:
    acc_best = max(rows, key=lambda r: r["accuracy"]); fast = min(rows, key=lambda r: r["latency_ms"]); small = min(rows, key=lambda r: r["size_mb"])
    facts += [f"Most accurate: <b>{acc_best['model']}</b> ({pct(acc_best['accuracy'])}).",
              f"Fastest: <b>{fast['model']}</b> ({fast['latency_ms']:.1f} ms); smallest: <b>{small['model']}</b> ({small['size_mb']:.1f} MB)."]
    if bert and q8:
        facts.append(f"INT8 student vs fp32 BERT-base: size ×{bert['size_mb']/q8['size_mb']:.1f} smaller, latency ×{bert['latency_ms']/q8['latency_ms']:.1f} faster, accuracy {100*(q8['accuracy']-bert['accuracy']):+.1f} pp.")
    if base and bert:
        facts.append(f"BERT-base vs tf-idf baseline: accuracy {100*(bert['accuracy']-base['accuracy']):+.1f} pp, OOS recall {100*(bert['oos_recall']-base['oos_recall']):+.1f} pp.")
    if res.get("typo"):
        t = res["typo"]; facts.append(f"Edit-distance correction ({t['n']} queries with injected typos, INT8 model): {pct(t['typos'])} → {pct(t['typos_corrected'])} (clean: {pct(t['clean'])}).")
else: facts.append("No results yet.")
y = 440
for f in facts: y = para(c, "•&nbsp;&nbsp;" + f, 500, y, 430, 12) - 8
c.showPage()

# 10 failures + limitations
slide(c, 10, "CONCLUSION", "Failure examples and limitations")
errs = (res["errors"].get(q8["model"], []) if res and q8 else [])
errs = sorted(errs, key=lambda e: -e["conf"])[:4]
d = [["Message (test set)", "True", "Predicted (INT8)", "Conf."]] + [[e["text"], e["gold"], e["pred"], f"{e['conf']:.2f}"] for e in errs]
if not errs: d.append(["-", "-", "-", "-"])
yb = table(c, d, 40, 455, [450, 150, 190, 60], 11)
bullets(c, [
    "<b>Proxy data:</b> CLINC150 is not real appliance tickets; the domain → team map is a design choice, not a measured quantity.",
    "<b>English only</b>; no Turkish evaluation. Out-of-scope is modelled as a 151st class, so OOS recall (table on slide 8) is the number to watch.",
    "<b>Latency</b> measured single-threaded on one CPU; other hardware changes absolute values, not the ranking.",
    "<b>Dynamic quantization</b> only covers linear layers; embeddings stay fp32.",
], y=yb - 20, size=13, gap=6)
c.showPage()

# 11 next
slide(c, 11, "CONCLUSION", "What I would do next")
bullets(c, [
    "Collect 50-100 real or hand-written domain tickets (Turkish and English) and fine-tune/evaluate on them.",
    "Replace the domain → team proxy with genuine team labels; tune the confidence threshold on a validation set for a target human-hand-off rate.",
    "Try ONNX Runtime export and quantizing embeddings as well; tune distillation α and T with a hyperparameter search.",
    "Use a multilingual encoder (e.g. XLM-R / BERTurk) for Turkish messages.",
    "Add monitoring: log low-confidence cases for active learning.",
], size=17)
c.showPage()

# 12 AI usage
slide(c, 12, "CONCLUSION", "AI usage statement")
para(c, cfg["ai_usage"], 50, 440, 860, 17)
c.showPage()
c.save()
print("wrote presentation.pdf", "(DRAFT - no results.json)" if DRAFT else "")
if unfilled: print("WARNING: fill these config.json fields before submitting:", ", ".join(unfilled))
