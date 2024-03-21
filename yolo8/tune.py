# The hyperparameter tuning is all local, without Comet ML, as it doesn't provide us
# with the ability to track the performance of the model during the tuning process.
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
    prog="Tune the hyperparameters of a custom YOLOv8 model",
    description="Script to tune the hyperparameters of a YOLOv8 model against a particular dataset",
)
parser.add_argument(
    "-m",
    "--model",
    help="""The relative file path to the YOLOv8 base model that will be used.
    If instead of a path it's [nano, small, medium, large, xlarge], the native YOLOv8 base model will be used.""",
    default="nano",
)
parser.add_argument(
    "-s",
    "--imgsz",
    help="The size of the training dataset images",
    default=640,
)
parser.add_argument(
    "-i",
    "--iterations",
    help="The number of tuning iterations",
    type=int,
    default=5,
)
parser.add_argument(
    "-e",
    "--epochs",
    help="The number of epochs to train on each iteration",
    type=int,
    default=50,
)
parser.add_argument(
    "-r",
    "--raytune",
    help="Use raytune on the tuning process",
    action="store_true"
)
naming_group = parser.add_mutually_exclusive_group()
naming_group.add_argument("-n", "--name", help="The name of the experiment")
naming_group.add_argument(
    "--dataname",
    help="A suffix added to the experiment name for easier identification of the dataset used",
)
args = parser.parse_args()

model_path = None
if args.model in YOLO8_MODELS:
    model_path = f"{YOLO8_MODELS[args.model]}.pt"
else:
    model_path = args.model

model = YOLO(model_path)
print("[INFO] Model has loaded")

experiment_name = None
if args.name:
    experiment_name = f"{args.name}"
else:
    metadata = [
        f"{YOLO8_MODELS.get(args.model, 'custom')}",  # The base model used
        f"i{args.iterations}",  # The number of tuning iterations
        f"e{args.epochs}",  # The number of epochs
    ]
    if args.raytune:
        metadata.append("ray") # If raytune was used
    experiment_name = "-".join(metadata)

# Tuning.
# Once the best hyperparams are found, we can load the `best_hyperparameters.yaml`
# file within the `tune` directories, and train that model with on `train.py`
print("[INFO] Start tuning")
results = model.tune(
    iterations=args.iterations,
    name=experiment_name,
    project="./tuning-models",
    data=os.path.abspath("./datasets/data.yaml"),
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=-1,  # Use auto batch size
    use_ray=args.raytune,
)
