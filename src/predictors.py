"""Uniform predictor wrappers: every model exposes predict_proba(list[str]) -> np.ndarray [n, 151]
and predict(text) -> (label, confidence). Inputs are normalized text (see textproc.normalize)."""
import json
import numpy as np
from .common import MODELS, get_labels
from .textproc import normalize


class Predictor:
    name = "base"
    def predict_proba(self, texts): raise NotImplementedError
    def predict(self, text):
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return self.labels[i], float(p[i])


class BaselinePredictor(Predictor):
    name = "tfidf+LR"
    def __init__(self, path=None):
        import joblib
        self.path = path or MODELS / "baseline.joblib"
        self.pipe = joblib.load(self.path)
        self.labels = list(self.pipe.classes_)
    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])
    def files(self): return [self.path]


class FFNPredictor(Predictor):
    name = "FFN+word2vec"
    def __init__(self, directory=None):
        import torch
        from gensim.models import KeyedVectors
        self.dir = directory or MODELS / "ffn"
        self.labels = json.load(open(self.dir / "labels.json"))
        self.kv = KeyedVectors.load(str(self.dir / "w2v.kv"))
        ckpt = torch.load(self.dir / "ffn.pt", map_location="cpu")
        from .train_ffn import make_ffn
        self.net = make_ffn(ckpt["dim"], ckpt["hidden"], len(self.labels))
        self.net.load_state_dict(ckpt["state"]); self.net.eval()
        self.torch = torch
    def embed(self, text):
        vs = [self.kv[w] for w in normalize(text).split() if w in self.kv]
        return np.mean(vs, axis=0) if vs else np.zeros(self.kv.vector_size, dtype="float32")
    def predict_proba(self, texts):
        x = self.torch.tensor(np.stack([self.embed(t) for t in texts]), dtype=self.torch.float32)
        with self.torch.no_grad():
            return self.torch.softmax(self.net(x), dim=-1).numpy()
    def files(self): return [self.dir / "ffn.pt", self.dir / "w2v.kv", self.dir / "labels.json"]


class TransformerPredictor(Predictor):
    """fp32 checkpoint directory (HF format) or an int8 dynamically-quantized one."""
    def __init__(self, directory, name, int8=False, max_len=48):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.name, self.max_len, self.int8, self.dir = torch, name, max_len, int8, directory
        self.tok = AutoTokenizer.from_pretrained(directory)
        cfg = AutoConfig.from_pretrained(directory)
        self.labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
        if int8:
            model = AutoModelForSequenceClassification.from_config(cfg)
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(torch.load(directory / "int8_state.pt", map_location="cpu", weights_only=False))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(directory)
        self.model = model.eval().to("cpu")
    def predict_proba(self, texts, batch=64):
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tok([normalize(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len, return_token_type_ids=False)
            with self.torch.no_grad():
                outs.append(self.torch.softmax(self.model(**enc).logits, dim=-1).numpy())
        return np.concatenate(outs)
    def weight_file(self):
        return self.dir / ("int8_state.pt" if self.int8 else "fp32_state.pt")
