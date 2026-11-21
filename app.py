"""Customer Support Ticket Router - Gradio app. Run: python app.py
Routing uses the INT8-quantized DistilBERT (R5); the other three models are shown side by side."""
import time
import gradio as gr
from src.common import MODELS, get_labels
from src.predictors import BaselinePredictor, FFNPredictor, TransformerPredictor
from src.teams import route, INTENT_DOMAIN, HUMAN
from src.textproc import build_lexicon, correct_text, eliza_reply, extract_codes, normalize
from src.common import load_splits

lexicon = build_lexicon(load_splits()["train"][0])
LADDER = []
def _try(label, build):
    try: LADDER.append((label, build()))
    except Exception as e: print(f"[warn] {label} not loaded: {e}")
_try("1 tf-idf + LogReg (fp32)", BaselinePredictor)
_try("2 word2vec + FFN (fp32)", FFNPredictor)
_try("3 BERT-base fine-tuned (fp32)", lambda: TransformerPredictor(MODELS / "bert_base", "BERT-base"))
_try("4 DistilBERT-KD INT8 (quantized)",
     lambda: TransformerPredictor(MODELS / "distilbert_kd_int8", "DistilBERT-KD-int8", int8=True))
QUANT = next((m for l, m in LADDER if l.startswith("4")), None)
if QUANT is None:
    raise SystemExit("Quantized model missing in models/. Run `python -m src.pull_models` or `python run_all.py`.")


def handle(message, threshold):
    message = (message or "").strip()
    if not message:
        return "Type a message first.", "", []
    reply = eliza_reply(message)
    if reply:
        return f"**Chatbot reply (rule-based):** {reply}", "greeting / small talk - no routing needed", []
    codes = extract_codes(message)
    norm = normalize(message)
    fixed, fixes = correct_text(norm, lexicon)
    prep = (f"normalized: `{norm}`\n\ncorrected: `{fixed}`" +
            (f"  (fixes: {', '.join(f'{a}→{b}' for a, b in fixes)})" if fixes else "") +
            (f"\n\nmodel code(s) found: **{', '.join(codes)}**" if codes else ""))
    rows, routed = [], None
    for label, m in LADDER:
        t0 = time.perf_counter(); intent, conf = m.predict(fixed); ms = (time.perf_counter() - t0) * 1000
        team = route(intent, conf, threshold)
        rows.append([label, intent, f"{conf:.2f}", team, f"{ms:.1f}"])
        if m is QUANT: routed = (intent, conf, team)
    intent, conf, team = routed
    head = (f"### Routed to: **{team}**\n\nIntent: `{intent}` · confidence {conf:.2f}" +
            (f" · model code {codes[0]}" if codes else "") +
            ("\n\n⚠️ low confidence or out-of-scope → human agent" if team == HUMAN else ""))
    return head, prep, rows


with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("## Customer Support Ticket Router\nType a support message. The **INT8-quantized "
                "DistilBERT** decides the route; the other models are shown side by side.")
    box = gr.Textbox(label="Customer message", placeholder="my dishwasher DW-6534 is not draining water", lines=2)
    thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence threshold (below → human agent)")
    btn = gr.Button("Route", variant="primary")
    out = gr.Markdown(); prep = gr.Markdown()
    tbl = gr.Dataframe(headers=["Model", "Intent", "Confidence", "Team", "ms"], interactive=False)
    gr.Examples([["my dishwasher DW-6534 is not draining water"], ["hello"],
                 ["how do i transfer money to my savings accout"], ["what is the capital of mongolia lol"],
                 ["i want to book a flight to paris"]], box)
    btn.click(handle, [box, thr], [out, prep, tbl]); box.submit(handle, [box, thr], [out, prep, tbl])

if __name__ == "__main__":
    demo.launch(share=True)
