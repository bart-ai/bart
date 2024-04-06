import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(
                    prog='Convert YOLO8 model to DNN model to be able to read it from OpenCV')
parser.add_argument('-m', '--model', help='The relative file path to the YOLO8 model')
args = parser.parse_args()

model = YOLO(args.model)

# export the model to ONNX format
model.export(format='onnx', imgsz=[640, 640])
