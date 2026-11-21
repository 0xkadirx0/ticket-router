"""Upload the trained models/ folder to your Hugging Face Hub model repo (run `huggingface-cli login` first)."""
import json
from huggingface_hub import HfApi
from .common import MODELS, ROOT

repo = json.load(open(ROOT / "config.json"))["hf_repo"]
assert repo and "FILL" not in repo, "Set hf_repo in config.json first"
api = HfApi(); api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path=str(MODELS), repo_id=repo)
print("uploaded to https://huggingface.co/" + repo)
