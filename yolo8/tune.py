# The hyperparameter tuning is all local, without Comet ML, as it doesn't provide us
# with the ability to track the performance of the model during the tuning process.
from ultralytics import YOLO
from ultralytics import settings
import os

# Set the datasets directory to the current working directory
settings.update({"datasets_dir": os.getcwd()})

print("[INFO] Starting...")

YOLO8_MODELS = {
    "nano": "yolov8n",
    "small": "yolov8s",
    "medium": "yolov8m",
    "large": "yolov8l",
    "xlarge": "yolov8x",
}

YOLO8_MODEL = YOLO8_MODELS["nano"]
model = YOLO(f"{YOLO8_MODEL}.pt")

print("[INFO] Model has loaded")

# Config
YOLO_DATA_YML_PATH = "./roboflow-billboards-yolo8-dataset/data.yaml"

# Hyperparameters
IMAGE_SIZE = 640
EPOCHS = 50
PATIENCE = 15
TUNING_ITERATIONS = 5

print("[INFO] Start tuning")

experiment_name = f"{YOLO8_MODEL}-e{EPOCHS}"

# Tuning.
# Once the best hyperparams are found, we can load the `best_hyperparameters.yaml`
# file within the `tune` directories, and train that model with on `train.py`

results = model.tune(
    iterations=TUNING_ITERATIONS,
    name=experiment_name,
    project="./",
    data=YOLO_DATA_YML_PATH,
    imgsz=IMAGE_SIZE,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch=-1,  # Use auto batch size
    # To enable training on Apple M1 and M2 chips,
    # you should specify 'mps' as your device when initiating the training process.
    # Comment out for automatic device selection
    device="mps",
)

