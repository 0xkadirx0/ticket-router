# 2-minute demo video: script (only the first 2:00 is watched)

Record the screen with voice narration (OBS, Zoom, or QuickTime). Say it in your own words; the lines below are a guide (~280 spoken words). Upload to YouTube (unlisted) or Drive ("anyone with the link"), then open the link in a private window.

**Before recording:** `python app.py` running; `presentation.pdf` open at the architecture slide; `results/results.md` or slide 8 ready; browser zoom ~125%.

| Time | On screen | Say |
|---|---|---|
| 0:00-0:15 | App home page | "Support teams get thousands of messages a day and each must reach the right team immediately. My app routes a message to a team, or to a human agent when it is unsure." |
| 0:15-0:35 | Type: `hello` | "A greeting gets a rule-based ELIZA reply, no model needed." |
| 0:35-0:55 | Type: `my dishwasher DW-6534 is not draining water` → Route | "It extracts the model code, normalizes the text, and the INT8-quantized DistilBERT predicts the intent with a confidence and a team. The table shows all four models and their latency side by side." |
| 0:55-1:15 | Difficult case: `how do i transfer money to my savings accout` (typo) and then `what is the capital of mongolia lol` | "Here a typo is fixed by minimum edit distance. And this out-of-scope question goes to a human agent because of the low confidence." |
| 1:15-1:30 | Architecture slide | "Pre-processing, four models with their precision, post-processing with a confidence threshold, and the Gradio UI; everything runs on CPU." |
| 1:30-1:45 | Results slide (table) | Read YOUR table: accuracy, size and latency of BERT-base vs the quantized student ("the quantized model is X times smaller and Y times faster for Z points of accuracy" - use your measured numbers). |
| 1:45-2:00 | Conclusion | "Quantization made the model cheaper with a small cost in accuracy; the weak spot is out-of-scope detection / my failure example. Next: real domain tickets and Turkish." |

Rules: new input typed live (not screenshots); voice on; stay under 2:00.
