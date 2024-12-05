import os
from dotenv import load_dotenv
import supervisely as sly
from src.florence2 import Florence2


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model = Florence2(
    use_gui=True,
    use_serving_gui_template=True,
    sliding_window_mode="none",
)
model.serve()
