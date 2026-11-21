"""Rung 1 - classical baseline: tf-idf (word 1-2 grams) + logistic regression."""
import argparse, joblib, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .common import MODELS, load_splits, set_seed
from .textproc import normalize


def main(limit=None):
    set_seed(); MODELS.mkdir(exist_ok=True)
    s = load_splits(limit)
    X = [normalize(t) for t in s["train"][0]]
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                         LogisticRegression(C=20, max_iter=1000))
    t0 = time.time(); pipe.fit(X, s["train"][1])
    va = pipe.score([normalize(t) for t in s["val"][0]], s["val"][1])
    print(f"baseline trained in {time.time()-t0:.1f}s | val acc {va:.4f}")
    joblib.dump(pipe, MODELS / "baseline.joblib", compress=3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); main(ap.parse_args().limit)
