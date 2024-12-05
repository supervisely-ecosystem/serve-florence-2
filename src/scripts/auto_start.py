import os

import supervisely as sly
from dotenv import load_dotenv
from supervisely.nn.utils import ModelSource, RuntimeType

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 69232  # <---- Change this to your task_id
method = "deploy_from_api"


# Pretrained
pretrained_model_data = {
    "deploy_params": {
        "model_files": {"checkpoint": "microsoft/Florence-2-base"},
        "model_source": ModelSource.PRETRAINED,
        "model_info": {
            "Model": "Florence-2-base",
            "Model Description": "Pretrained model with FLD-5B",
            "Model Size": "0.23B",
            "COCO Det. val2017 mAP": 34.7,
            "meta": {
                "task_type": "prompt-based object detection",
                "model_name": "Florence-2-base",
                "model_files": {"checkpoint": "microsoft/Florence-2-base"},
            },
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}
api.app.send_request(task_id, method, pretrained_model_data)
