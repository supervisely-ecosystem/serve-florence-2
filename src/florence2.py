import os
from typing import Any, Dict, List, Tuple

import numpy as np
import supervisely as sly
import torch
import torchvision.transforms as T
from huggingface_hub import list_repo_tree, snapshot_download
from PIL import Image
from supervisely.app.widgets import Checkbox, Field
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType
from supervisely.nn.inference.inference import (
    get_hardware_info,
    get_name_from_env,
    logger,
)
from supervisely.nn.prediction_dto import PredictionBBox
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2(sly.nn.inference.PromptBasedObjectDetection):
    FRAMEWORK_NAME = "Florence 2"
    MODELS = "src/models.json"
    APP_OPTIONS = "src/app_options.yaml"
    INFERENCE_SETTINGS = "src/inference_settings.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def initialize_extra_widgets(gui_instance):
            gui_instance.update_pretrained_checkbox = Checkbox("Update pretrained model", False)
            update_pretrained_field = Field(
                gui_instance.update_pretrained_checkbox,
                "Update Model",
                "If checked, the model will be downloaded from HuggingFace even if it exists in cache",
            )
            return [update_pretrained_field]

        self._gui._initialize_extra_widgets = lambda: initialize_extra_widgets(self._gui)
        base_widgets = self._gui._initialize_layout()
        extra_widgets = self._gui._initialize_extra_widgets()
        self._gui.widgets = base_widgets + extra_widgets
        self._gui.card = self._gui._get_card()
        self._user_layout = self._gui.widgets
        self._user_layout_card = self._gui.card

        self.update_pretrained = False
        if hasattr(self._gui, "update_pretrained_checkbox"):
            self.update_pretrained = self._gui.update_pretrained_checkbox.is_checked()

        original_on_serve_callbacks = self.gui.on_serve_callbacks.copy()
        self.gui.on_serve_callbacks = []

        def on_serve_with_checkbox_check(gui):
            if hasattr(gui, "update_pretrained_checkbox"):
                self.update_pretrained = gui.update_pretrained_checkbox.is_checked()
                logger.debug(f"On Serve Callback - Update pretrained model: {self.update_pretrained}")

        self.gui.on_serve_callbacks.append(on_serve_with_checkbox_check)
        self.gui.on_serve_callbacks.extend(original_on_serve_callbacks)

        # disable GUI widgets
        self.gui.set_project_meta = self.set_project_meta
        self.weights_cache_dir = self._checkpoints_cache_dir()
        self.default_task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.torch_dtype = torch.float32

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
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
        self.task_prompt = settings.get("task_prompt", self.default_task_prompt)
        size_scaler = None
        img_input = Image.open(image_path)
        # TODO uncomment next line if you need to resize the image to optimize memory usage
        # img_input, size_scaler = self._prepare_input(image_path)
        mapping = settings.get("mapping")
        text = settings.get("text")

        # 2. Inference Classes Mapping
        if mapping is not None and text is None:
            predictions_mapping = self._classes_mapping_inference(img_input, mapping)
            # 3. Postprocess
            predictions = self._format_predictions_cm(predictions_mapping, size_scaler)
        elif mapping is None and text is not None:
            predictions_mapping = self._common_prompt_inference(img_input, text)
            # 3. Postprocess
            predictions = self._format_predictions_cp(predictions_mapping, size_scaler)
        else:
            raise ValueError("Either 'mapping' or 'text' should be provided")

        return predictions

    def predict_batch(self, images_np, settings):
        task_prompt = settings.get("task_prompt", self.default_task_prompt)
        if task_prompt in [
            "<CAPTION>+<CAPTION_TO_PHRASE_GROUNDING>",
            "<DETAILED_CAPTION>+<CAPTION_TO_PHRASE_GROUNDING>",
            "<MORE_DETAILED_CAPTION>+<CAPTION_TO_PHRASE_GROUNDING>",
        ]:
            self.task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            cascaded_task = True
        else:
            self.task_prompt = task_prompt
            cascaded_task = False

        text = settings.get("text", "find all objects")
        mapping = settings.get("mapping")

        images = [Image.fromarray(img) for img in images_np]

        if mapping is None:
            if cascaded_task:
                caption_prompt = task_prompt.split("+")[0]
                texts = self.batch_inference(images, caption_prompt)
                texts = [txt[caption_prompt] for txt in texts]
                parsed_answers = self.batch_inference(
                    images, "<CAPTION_TO_PHRASE_GROUNDING>", texts
                )
            else:
                parsed_answers = self.batch_inference(images, self.task_prompt, text)

            batch_predictions = []
            for answer in parsed_answers:
                predictions = self._format_predictions_cp(answer, size_scaler=None)
                batch_predictions.append(predictions)

        elif mapping is not None and text is None:
            batch_predictions = []
            for image in images:
                predictions_mapping = self._classes_mapping_inference(image, mapping)
                predictions = self._format_predictions_cm(predictions_mapping, size_scaler=None)
                batch_predictions.append(predictions)

        return batch_predictions

    def batch_inference(self, images, task_prompt, text=None):
        if task_prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
            if isinstance(text, str):
                prompt = [task_prompt + text] * len(images)
            elif isinstance(text, list):
                prompt = [task_prompt + txt for txt in text]
        else:
            prompt = [task_prompt] * len(images)

        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(
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
        parsed_answers = [
            self.processor.post_process_generation(
                text, task=task_prompt, image_size=(img.width, img.height)
            )
            for text, img in zip(generated_texts, images)
        ]
        return parsed_answers

    def _common_prompt_inference(self, img_input: Image.Image, text: str):
        if text == "":
            text = self._get_detailed_caption_text(img_input)
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
        return parsed_answer

    def _classes_mapping_inference(self, img_input: Image.Image, mapping: dict):
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
        return predictions_mapping

    def _get_detailed_caption_text(self, img_input: Image.Image) -> str:
        logger.info("Text prompt is empty. Getting detailed caption for the image...")
        task_prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=img_input, return_tensors="pt").to(
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
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(img_input.width, img_input.height),
        )
        detailed_caption_text = parsed_answer[task_prompt]
        logger.info(f"Got detailed caption: {detailed_caption_text}")
        return detailed_caption_text

    def _prepare_input(self, image_path: str):
        image = Image.open(image_path)
        new_size = self._get_resized_dimensions(image.size)
        size_scaler = (image.width / new_size[0], image.height / new_size[1])
        image_resized = image.resize(new_size)
        return image_resized, size_scaler

    def _format_prediction(
        self,
        prediction: dict,
        class_name: str,
        size_scaler: Tuple[int, int] = None,
    ) -> List[PredictionBBox]:
        scaled_bboxes = []
        bboxes = prediction[self.task_prompt]["bboxes"]
        scaled_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if size_scaler is not None:
                x1, y1, x2, y2 = (
                    round(x1 * float(size_scaler[0])),
                    round(y1 * float(size_scaler[1])),
                    round(x2 * float(size_scaler[0])),
                    round(y2 * float(size_scaler[1])),
                )
            else:
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            bbox_yxyx = [y1, x1, y2, x2]
            pred_box = PredictionBBox(class_name, bbox_yxyx, None)
            scaled_bboxes.append(pred_box)
        return scaled_bboxes

    def _format_predictions_cm(
        self, predictions_mapping: dict, size_scaler: List
    ) -> List[List[PredictionBBox]]:
        """For classes mapping"""
        postprocessed_preds = []
        for class_name, prediction in predictions_mapping.items():
            scaled_bboxes = self._format_prediction(prediction, class_name, size_scaler)
            postprocessed_preds.extend(scaled_bboxes)
        return postprocessed_preds

    def _format_predictions_cp(
        self,
        predictions: dict,
        size_scaler: Tuple[int, int] = None,
    ) -> List[List[PredictionBBox]]:
        """For common prompt"""
        postprocessed_preds = []
        bboxes = predictions[self.task_prompt]["bboxes"]
        class_names = predictions[self.task_prompt]["labels"]
        for class_name, bbox in zip(class_names, bboxes):
            x1, y1, x2, y2 = bbox
            if size_scaler is not None:
                x1, y1, x2, y2 = (
                    round(x1 * float(size_scaler[0])),
                    round(y1 * float(size_scaler[1])),
                    round(x2 * float(size_scaler[0])),
                    round(y2 * float(size_scaler[1])),
                )
            else:
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            if not y1 > y2 and not x1 > x2:
                bbox_yxyx = [y1, x1, y2, x2]
                if not class_name:
                    class_name = "object"
                pred_box = PredictionBBox(class_name, bbox_yxyx, None)
                postprocessed_preds.append(pred_box)
        return postprocessed_preds

    def _download_pretrained_model(self, model_files: dict):
        """
        Diff to :class:`Inference`:
           - The model is downloaded from the Hugging Face
        """
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
        if self.update_pretrained:
            logger.debug(f"Downloading {repo_id} to {local_model_path}...")
            files_info = list_repo_tree(repo_id)
            total_size = sum([file.size for file in files_info])
            with self.gui._download_progress(
                message=f"Updating: '{model_name}'",
                total=total_size,
                unit="bytes",
                unit_scale=True,
            ) as download_pbar:
                self.gui.download_progress.show()
                snapshot_download(repo_id=repo_id, local_dir=local_model_path)
                download_pbar.update(total_size)
                # TODO update progress widget to use async_tqdm
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

    def _create_label(self, dto: PredictionBBox) -> sly.Label:
        """
        Create a label from the prediction DTO.
        Diff to :class:`ObjectDetection`:
              - class_name is appended with "_bbox" to match the class name in the project
        """
        class_name = dto.class_name + "_bbox"
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Rectangle)
            )
            obj_class = self.model_meta.get_obj_class(class_name)
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label

    def set_project_meta(self, inference):
        """The model does not have predefined classes.
        In case of prompt-based models, the classes are defined by the user."""
        self.gui._model_classes_widget_container.hide()
        return

    def _get_resized_dimensions(self, orig_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Resize image to the maximum size of 224x224 to optimize memoru usage
        while preserving the aspect ratio.
        If one of the dimensions will be less than 24 after resizing, it will be set to 24.
        So the other dimension will be adjusted accordingly and might be bigger than 224, but this is fine.
        """
        max_size = max(self.img_size[0], self.img_size[1])
        width, height = orig_size
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        if new_width < 24:
            new_width = 24
            new_height = int(24 * height / width)
        if new_height < 24:
            new_height = 24
            new_width = int(24 * width / height)
        return (new_width, new_height)
