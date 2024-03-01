import argparse

from PIL import Image
from ultralytics import YOLO

parser = argparse.ArgumentParser(
                    prog='Validate YOLOv8 Model',
                    description='Script to validate a pretrained YOLOv8n against a particular dataset')
parser.add_argument('-d', '--data', help='The relative file path for YOLOv8 data set to validate')
parser.add_argument('-m', '--model', help='The relative file path to the YOLOv8 model that will be used to predict')
args = parser.parse_args()

modelPath = args.model
dataSet = args.data

# Load a pretrained YOLOv8n model
model = YOLO(modelPath)

validation_results = model.val(
  data=dataSet,
  imgsz=640,
  batch=16,
  conf=0.8,
  iou=0.6,
  # device="mps",
)
