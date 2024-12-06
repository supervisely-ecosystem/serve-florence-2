import json
import os

from huggingface_hub import snapshot_download

STORAGE_DIR = "/root/.cache/supervisely/checkpoints"


def download_models(model_files: dict, storage_dir: str):
    print(f"Checking if storage directory {storage_dir} exists...")
    if not os.path.exists(storage_dir):
        print(f"Creating storage directory {storage_dir}...")
        os.makedirs(storage_dir, exist_ok=True)
    for file in model_files:
        repo_id = file["meta"]["model_files"]["checkpoint"]
        model_name = repo_id.split("/")[1]
        local_model_path = f"{storage_dir}/{model_name}"
        print(f"Downloading {repo_id} to {local_model_path}...")
        snapshot_download(repo_id=repo_id, local_dir=local_model_path)
        print(f"Downloaded {repo_id}")


def load_models(json_file_path: str):
    print(f"Loading models from {json_file_path}...")
    with open(json_file_path, "r") as json_file:
        model_files = json.load(json_file)
    print(f"Loaded models: {model_files}")
    return model_files


try:
    print("Starting model download process...")
    model_files = load_models("models.json")
    download_models(model_files, STORAGE_DIR)
except Exception as e:
    print("Something went wrong while downloading the models", repr(e))
else:
    downloaded = os.listdir(STORAGE_DIR)
    print("Florence 2 models downloaded successfully:", downloaded)
