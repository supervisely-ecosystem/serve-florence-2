import os
from typing import Any, Dict, List, Tuple

import numpy as np
import supervisely as sly
import torch
import torchvision.transforms as T
from huggingface_hub import list_repo_tree, snapshot_download
from PIL import Image
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
from supervisely.nn.inference.inference import (
    get_hardware_info,
    get_name_from_env,
    logger,
)
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
        # self.weights_cache_dir = "/root/.cache/supervisely/checkpoints"
        self.weights_cache_dir = "/home/serpntns/Work/serve-florence-2/app_data/models"
        self.task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.torch_dtype = torch.float32

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

        h, w = 224, 224
        self.img_size = [w, h]

        if runtime == RuntimeType.PYTORCH:
            model_path = model_files["checkpoint"]
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=self.torch_dtype, trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = self.model.to(device)

    def predict(self, image_path: np.ndarray, settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(image_path, settings)

    @torch.no_grad()
    def _predict_pytorch(self, image_path: str, settings: dict = None) -> List[PredictionBBox]:
        # 1. Preprocess
        img_input, size_scaler = self._prepare_input(image_path)
        # 2. Inference
        mapping = settings.get("mapping", {})
        predictions_mapping = {}
        for target_class, text in mapping.items():
            if text is None:
                prompt = self.task_prompt
            else:
                prompt = self.task_prompt + text
            inputs = self.processor(text=prompt, images=img_input, return_tensors="pt").to(
                self.device, self.torch_dtype
            )
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
            parsed_answer = self.processor.post_process_generation(
                generated_texts[0],
                task=self.task_prompt,
                image_size=(img_input.width, img_input.height),  # W, H
            )
            predictions_mapping.update({target_class: parsed_answer})

        # 3. Postprocess
        predictions = self._format_predictions(predictions_mapping, size_scaler)

        return predictions

    def _prepare_input(self, image_path: str, device=None):
        image = Image.open(image_path)
        new_size = self._get_resized_dimensions(image.size)
        size_scaler = (image.width / new_size[0], image.height / new_size[1])
        image_resized = image.resize(new_size)
        return image_resized, size_scaler

    def _format_prediction(
        self,
        prediction: dict,
        class_name: str,
        size_scaler: Tuple[int, int],
    ) -> List[PredictionBBox]:
        scaled_bboxes = []
        bboxes = prediction[self.task_prompt]["bboxes"]
        scaled_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (
                round(x1 * float(size_scaler[0])),
                round(y1 * float(size_scaler[1])),
                round(x2 * float(size_scaler[0])),
                round(y2 * float(size_scaler[1])),
            )
            bbox_yxyx = [y1, x1, y2, x2]
            pred_box = PredictionBBox(class_name, bbox_yxyx, None)
            scaled_bboxes.append(pred_box)
        return scaled_bboxes

    def _format_predictions(
        self, predictions_mapping: dict, size_scaler: List
    ) -> List[List[PredictionBBox]]:
        postprocessed_preds = []
        for class_name, prediction in predictions_mapping.items():
            scaled_bboxes = self._format_prediction(prediction, class_name, size_scaler)
            postprocessed_preds.extend(scaled_bboxes)
        return postprocessed_preds

    def _download_pretrained_model(self, model_files: dict):
        if os.path.exists(self.weights_cache_dir):
            cached_weights = os.listdir(self.weights_cache_dir)
            logger.debug(f"Cached_weights: {cached_weights}")
        else:
            logger.debug(
                f"Directory {self.weights_cache_dir} does not exist. Downloading weights..."
            )
        repo_id = model_files["checkpoint"]
        model_name = repo_id.split("/")[1]
        local_model_path = f"{self.weights_cache_dir}/{model_name}"
        logger.debug(f"Downloading {repo_id} to {local_model_path}...")
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

    def _create_label(self, dto: PredictionBBox):
        class_name = dto.class_name + "_bbox"
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Rectangle)
            )
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label

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

    def _get_resized_dimensions(self, orig_size: Tuple[int, int]) -> Tuple[int, int]:
        max_size = max(self.img_size[0], self.img_size[1])
        width, height = orig_size
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        return (new_width, new_height)
