"""Rung 4 - post-training dynamic INT8 quantization of every nn.Linear layer (PyTorch)."""
import argparse, shutil, torch
from transformers import AutoModelForSequenceClassification
from .common import MODELS, size_mb


def main(src, dst):
    src, dst = MODELS / src, MODELS / dst
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():                       # config + tokenizer files (small)
        if f.suffix in {".json", ".txt", ".model"}:
            shutil.copy(f, dst / f.name)
    torch.save(q.state_dict(), dst / "int8_state.pt")
    print(f"fp32 {size_mb(src / 'fp32_state.pt'):.1f} MB -> int8 {size_mb(dst / 'int8_state.pt'):.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="distilbert_kd"); ap.add_argument("--dst", default="distilbert_kd_int8")
    a = ap.parse_args(); main(a.src, a.dst)
