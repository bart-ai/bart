from comet_ml import Experiment
from ultralytics import YOLO
from ultralytics import settings
import os

# Set the datasets directory to the current working directory
settings.update({"datasets_dir": os.getcwd()})

print("[INFO] Starting...")

# This is for tracking the model's performance over time.
# You should have the comet API key defined in your environment variables.
# export COMET_API_KEY=<Your API Key>
experiment = Experiment(
    project_name="test-yolov8-billboards", api_key=os.environ["COMET_API_KEY"]
)

print('[INFO] Comet ML configured')

YOLO8_MODELS = {
    "nano": "yolov8n",
    "small": "yolov8s",
    "medium": "yolov8m",
    "large": "yolov8l",
    "xlarge": "yolov8x",
}

YOLO8_MODEL = YOLO8_MODELS["nano"]
model = YOLO(f"{YOLO8_MODEL}.pt")

print('[INFO] Model has loaded')

# Config
YOLO_DATA_YML_PATH = './roboflow-billboards-yolo8-dataset/data.yaml'

# Hyperparameters
IMAGE_SIZE=640
EPOCHS=50
PATIENCE=15

print('[INFO] Start traning')

experiment_name = f"{YOLO8_MODEL}-e{EPOCHS}"
experiment.set_name(experiment_name)

# Training.
results = model.train(
   name=experiment_name,
   project='./',
   data=YOLO_DATA_YML_PATH,
   imgsz=IMAGE_SIZE,
   epochs=EPOCHS,
   patience=PATIENCE,
   batch=-1,  # Use auto batch size

  # Load the hyperparameters from `tune.py`
  # cfg="best_hyperparameters.yaml"

  # To enable training on Apple M1 and M2 chips,
  # you should specify 'mps' as your device when initiating the training process.
  # Comment out for automatic device selection
  device="mps",
)

# This exports the model in an onnx format, which is later used in the wb app.
# You can also export an exisiting model result using the convert_to_onnx.py script.
model.export(format='onnx', simplify=True, imgsz=[IMAGE_SIZE, IMAGE_SIZE])
