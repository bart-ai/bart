import os

import cv2
import numpy as np

import cvutils

IMAGE_SIZE = 640

def get_model():
    cwd = os.path.dirname(os.path.realpath(__file__))
    net = cv2.dnn.readNet(
        # YOLOv8 ONNX Model
        f"{cwd}/model/billboard-detection/best.onnx",
        # YOLOv8 Pre-Trained Model
        # f"{cwd}/model/billboard-detection/yolov8n.onnx",
        # Caffe Model
        # f"{cwd}/model/face-detection/res10_300x300_ssd_iter_140000.caffemodel",
        # f"{cwd}/model/face-detection/deploy.prototxt",
    )
    dimensions = (IMAGE_SIZE, IMAGE_SIZE)
    model = (net, dimensions)
    return model

def detect(frame, confidence=0.8, model=get_model()):
    # Feed the frame to the model
    net, dimensions = model

    # Read the input image
    original_image = frame
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=dimensions, swapRB=True)
    net.setInput(blob)

    # Perform inference
    outputs = net.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= confidence:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        startX = round(box[0] * scale)
        startY = round(box[1] * scale)
        endX = round((box[0] + box[2]) * scale)
        endY = round((box[1] + box[3]) * scale)
        cvutils.draw(original_image, (startX, startY, endX, endY), f"Billboard ({scores[index]:.2f})")

    return frame
