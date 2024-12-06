import json
import os

from huggingface_hub import snapshot_download

STORAGE_DIR = os.path.expanduser("~/.cache/supervisely/checkpoints")


def download_models(model_files: dict, storage_dir: str):
    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)
    for file in model_files:
        repo_id = file["checkpoint"]
        model_name = repo_id.split("/")[1]
        print(f"Downloading {repo_id} to {local_model_path}...")
        local_model_path = f"{storage_dir}/{model_name}"
        snapshot_download(repo_id=repo_id, local_dir=local_model_path)
        print(f"Downloaded {repo_id}")


def load_models(json_file_path: str):
    with open(json_file_path, "r") as json_file:
        model_files = json.load(json_file)
    return model_files


if __name__ == "__main__":
    try:
        model_files = load_models("models.json")
        download_models(model_files, STORAGE_DIR)
    except Exception as e:
        print("Something went wrong while downloading the models", e)
    else:
        downloaded = os.listdir(STORAGE_DIR)
        print("Florence 2 models downloaded successfully:", downloaded)
