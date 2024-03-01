import argparse
from comet_ml import Experiment
from ultralytics import YOLO
from ultralytics import settings
import os
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(
                    prog='Train a custom YOLOv8 model',
                    description='Script to train a custom (already trained) YOLOv8n against a particular dataset')
parser.add_argument('-d', '--data', help='The relative file path for YOLOv8 data set')
parser.add_argument('-m', '--model', help='The relative file path to the YOLOv8 model that will be used to extend training')
parser.add_argument('-n', '--name', help='The name of the experiment')
args = parser.parse_args()

modelPath = args.model
dataSetPath = args.data
experimentName = args.name

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

# Load the pre-trained YOLOv8 model.
model = YOLO(modelPath)

print('[INFO] Model has loaded')

# Hyperparameters
IMAGE_SIZE=640
EPOCHS=50

print('[INFO] Start traning')

experiment_name = f"{experimentName}-e{EPOCHS}"
experiment.set_name(experiment_name)

# Training.
results = model.train(
   name=experiment_name,
   project='./',
   data=dataSetPath,
   imgsz=IMAGE_SIZE,
   epochs=EPOCHS,
   batch=-1,  # Use auto batch size

  # Load the hyperparameters from `tune.py`
  # cfg="best_hyperparameters.yaml"

  # To enable training on Apple M1 and M2 chips,
  # you should specify 'mps' as your device when initiating the training process.
  # Comment out for automatic device selection
  # device="mps",
)

# This exports the model in an onnx format, which is later used in the wb app.
# You can also export an exisiting model result using the convert_to_onnx.py script.
model.export(format='onnx', simplify=True, imgsz=[IMAGE_SIZE, IMAGE_SIZE])
