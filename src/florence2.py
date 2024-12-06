import os
from typing import Any, Dict, List, Tuple

import numpy as np
import supervisely as sly
import torch
import torchvision.transforms as T
from huggingface_hub import list_repo_tree, snapshot_download
from PIL import Image
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
from supervisely.nn.inference.inference import get_hardware_info, get_name_from_env
from supervisely.nn.prediction_dto import PredictionBBox
from transformers import AutoModelForCausalLM, AutoProcessor

SERVE_PATH = "src"


class Florence2(sly.nn.inference.PromptBasedObjectDetection):
    FRAMEWORK_NAME = "Florence 2"
    MODELS = "src/models.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # disable GUI widgets
        self.gui.set_project_meta = self.set_project_meta
        self.gui.set_inference_settings = self.set_inference_settings
        self.weights_cache_dir = os.path.expanduser("~/.cache/supervisely/checkpoints")

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        checkpoint_path = model_files["checkpoint"]
        if model_source == ModelSource.PRETRAINED:
            self.classes = []
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
                model_source=model_source,
            )

        h, w = 640, 640
        self.img_size = [w, h]
        self.transforms = T.Compose(
            [
                T.Resize((h, w)),
                T.ToTensor(),
            ]
        )

        if runtime == RuntimeType.PYTORCH:
            model_path = model_files["checkpoint"]
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = self.model.to(device)

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)

    @torch.no_grad()
    def _predict_pytorch(
        self, images_np: List[np.ndarray], settings: dict = None
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            img_input, size_input, orig_target_sizes = self._prepare_input(images_np)
        # 2. Inference
        with Timer() as inference_timer:
            outputs = self.model(img_input)
        # 3. Postprocess
        with Timer() as postprocess_timer:
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
            labels, boxes, scores = labels.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()
            predictions = self._format_predictions(labels, boxes, scores, settings)
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _prepare_input(self, images_np: List[np.ndarray], device=None):
        if device is None:
            device = self.device
        imgs_pil = [Image.fromarray(img) for img in images_np]
        orig_sizes = torch.as_tensor([img.size for img in imgs_pil])
        img_input = torch.stack([self.transforms(img) for img in imgs_pil])
        size_input = torch.tensor([self.img_size * len(images_np)]).reshape(-1, 2)
        return img_input.to(device), size_input.to(device), orig_sizes.to(device)

    def _format_prediction(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_tresh: float
    ) -> List[PredictionBBox]:
        predictions = []
        for label, bbox_xyxy, score in zip(labels, boxes, scores):
            if score < conf_tresh:
                continue
            class_name = self.classes[label]
            bbox_xyxy = np.round(bbox_xyxy).astype(int)
            bbox_xyxy = np.clip(bbox_xyxy, 0, None)
            bbox_yxyx = [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]]
            bbox_yxyx = list(map(int, bbox_yxyx))
            predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
        return predictions

    def _format_predictions(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, settings: dict
    ) -> List[List[PredictionBBox]]:
        thres = settings["confidence_threshold"]
        predictions = [self._format_prediction(*args, thres) for args in zip(labels, boxes, scores)]
        return predictions

    def _download_pretrained_model(self, model_files: dict):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        repo_id = model_files["checkpoint"]
        model_name = repo_id.split("/")[1]
        local_model_path = f"{self.weights_cache_dir}/{model_name}"
        files_info = list_repo_tree(repo_id)
        total_size = sum([file.size for file in files_info])
        with self.gui._download_progress(
            message=f"Downloading: '{model_name}'",
            total=total_size,
            unit="bytes",
            unit_scale=True,
        ) as download_pbar:
            self.gui.download_progress.show()
            snapshot_download(repo_id=repo_id, local_dir=local_model_path)
            # TODO update widget
        return {"checkpoint": local_model_path}

    def _load_model_headless(
        self,
        model_files: dict,
        model_source: str,
        model_info: dict,
        device: str,
        runtime: str,
        **kwargs,
    ):
        """
        Diff to :class:`Inference`:
           - _set_model_meta_from_classes() removed due to lack of classes
        """
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
            **kwargs,
        }
        if model_source == ModelSource.CUSTOM:
            self._set_model_meta_custom_model(model_info)
            self._set_checkpoint_info_custom_model(deploy_params)
        self._load_model(deploy_params)

    def _load_model(self, deploy_params: dict):
        """
        Diff to :class:`Inference`:
           - self.model_precision replaced with the cuda availability check
        """
        self.model_source = deploy_params.get("model_source")
        self.device = deploy_params.get("device")
        self.runtime = deploy_params.get("runtime", RuntimeType.PYTORCH)
        self.model_precision = torch.float16 if torch.cuda.is_available() else torch.float32
        self._hardware = get_hardware_info(self.device)
        self.load_model(**deploy_params)
        self._model_served = True
        self._deploy_params = deploy_params
        if self.gui is not None:
            self.update_gui(self._model_served)
            self.gui.show_deployed_model_info(self)

    def get_info(self) -> Dict[str, Any]:
        return {
            "app_name": get_name_from_env(default="Neural Network Serving"),
            "session_id": self.task_id,
            "task type": "prompt-based object detection",
            "sliding_window_support": self.sliding_window_mode,
            "batch_inference_support": self.is_batch_inference_supported(),
        }

    def set_project_meta(self, inference):
        self.gui._model_classes_widget_container.hide()
        return

    def set_inference_settings(self, inference):
        self.gui._model_inference_settings_container.hide()
        return
