from comet_ml import Experiment
from ultralytics import YOLO
from ultralytics import settings
import os
import argparse

YOLO8_MODELS = {
    "nano": "yolov8n",
    "small": "yolov8s",
    "medium": "yolov8m",
    "large": "yolov8l",
    "xlarge": "yolov8x",
}

# Set the datasets directory to the current working directory
settings.update({"datasets_dir": os.getcwd()})

print("[INFO] Starting...")

parser = argparse.ArgumentParser(
    prog="Train a custom YOLOv8 model",
    description="Script to train a YOLOv8 model against a particular dataset",
)
parser.add_argument(
    "-m",
    "--model",
    help="""The relative file path to the YOLOv8 base model that will be used to extend training. 
    If instead of a path it's [nano, small, medium, large, xlarge], the native YOLOv8 base model will be used.""",
    default="nano"
)
parser.add_argument(
    "-s",
    "--imgsz",
    help="The size of the training dataset images",
    default=640,
)
parser.add_argument(
    "-e",
    "--epochs",
    help="The number of epochs to train the model",
    type=int,
    default=50,
)
parser.add_argument(
    "-hp",
    "--hyperparams",
    help="The relative file path to the hyperparameters file",
)
parser.add_argument("-n", "--name", help="The name of the experiment")
args = parser.parse_args()


# This is for tracking the model's performance over time.
# You should have the comet API key defined in your environment variables.
# export COMET_API_KEY=<Your API Key>
experiment = Experiment(
    project_name="test-yolov8-billboards", api_key=os.environ["COMET_API_KEY"]
)

print('[INFO] Comet ML configured')

model_path = None
if args.model in YOLO8_MODELS:
    model_path = f"{YOLO8_MODELS[args.model]}.pt"
else:
    model_path = args.model

model = YOLO(model_path)
print('[INFO] Model has loaded')

experiment_name = None
if args.name:
    experiment_name = f"{args.name}"
else:
    experiment_name = f"{YOLO8_MODELS.get(args.model, 'custom')}-e{args.epochs}"

experiment.set_name(experiment_name)

print("[INFO] Start traning")

# Training.
results = model.train(
    name=experiment_name,
    project="./models",
    data="./datasets/data.yaml",
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=-1,  # Use auto batch size
    cfg=args.hyperparams if args.hyperparams else None, # Load the hyperparameters from `tune.py`
)

# This exports the model in an onnx format, which is later used in the web app.
# You can also export an exisiting model result using the convert_to_onnx.py script.
model.export(format='onnx', simplify=True, imgsz=[args.imgsz, args.imgsz])
