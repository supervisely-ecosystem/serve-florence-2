import os

import numpy as np
from PIL import Image
from supervisely.nn import ModelSource, RuntimeType

from src.florence2 import Florence2

os.environ["SLY_APP_DATA_DIR"] = "app_data"

model = Florence2()

model_info = model.pretrained_models[0]

model._load_model_headless(
    model_files={
        "checkpoint": os.path.expanduser(
            "~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
        ),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.PYTORCH,
)

image = Image.open("src/scripts/sample.jpg").convert("RGB")
img = np.array(image)

ann = model._inference_auto([img], {"confidence_threshold": 0.5})[0][0]

ann.draw_pretty(img)
Image.fromarray(img).save("src/scripts/predict.jpg")
