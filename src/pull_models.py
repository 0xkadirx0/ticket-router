"""Download trained model files from the Hugging Face Hub (weights are not stored in git).
Usage: python -m src.pull_models  (reads hf_repo from config.json)"""
import json
from huggingface_hub import snapshot_download
from .common import MODELS, ROOT

repo = json.load(open(ROOT / "config.json"))["hf_repo"]
assert repo and "FILL" not in repo, "Set hf_repo in config.json first"
snapshot_download(repo_id=repo, local_dir=MODELS)
print("models downloaded to", MODELS)
