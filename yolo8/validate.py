import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser(
                    prog='Validate YOLOv8 Model',
                    description='Script to validate a pretrained YOLOv8n against a particular dataset')
parser.add_argument('-m', '--model', help='The relative file path to the YOLOv8 model that will be used to predict')
args = parser.parse_args()

modelPath = args.model

# Load a pretrained YOLOv8n model
model = YOLO(modelPath)

validation_results = model.val(
  data="./datasets/data.yaml",
  imgsz=640,
  batch=16,
  conf=0.8,
  iou=0.6,
  # device="mps",
)
