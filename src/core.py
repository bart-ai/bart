import os

import cv2
import cvutils
import numpy as np

IMAGE_SIZE = 640

def get_model():
    cwd = os.path.dirname(os.path.realpath(__file__))
    net = cv2.dnn.readNet(
        # YOLOv8 ONNX Model
        # f"{cwd}/model/billboard-detection/best.onnx",
        # YOLOv8 Pre-Trained Model
        f"{cwd}/model/billboard-detection/yolov8n.onnx",
        # Caffe Model
        # f"{cwd}/model/face-detection/res10_300x300_ssd_iter_140000.caffemodel",
        # f"{cwd}/model/face-detection/deploy.prototxt",
    )
    dimensions = (IMAGE_SIZE, IMAGE_SIZE)
    model = (net, dimensions)
    return model

def detect(frame, confidence=0.2, model=get_model()):
    # Feed the frame to the model
    net, net_dimensions = model
    blob = cv2.dnn.blobFromImage(frame, 1.0, net_dimensions)
    net.setInput(blob)

    # Run the model itself
    detections = net.forward()

    billboardDetection = detections[0]
    print('First detection', billboardDetection[0:5, 0])
    print('\nSTART FRAME\n')
    # mostly taken from https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
    # loop over the detections
    for i in range(billboardDetection.shape[1]):
        current_detection = billboardDetection[0:5, i]
        # print('current_detection', current_detection)
        # extract the confidence associated with the prediction
        detection_confidence = current_detection[4]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if not detection_confidence > confidence:
            continue
        print('\ndetection_confidence', detection_confidence)

        # compute the (x, y)-coordinates of the bounding box for the object
        # multiply by the frame size to get the actual coordinates
        height, width = frame.shape[:2]
        # print('height', height)
        # print('width', width)

        (startX, startY, endX, endY) = (
            current_detection[0:4] * np.array([width, height, width, height])
        ).astype("int")
        print('startX', startX)
        print('startY', startY)
        print('endX', endX)
        print('endY', endY)
        print('\n')

        # draw the bounding box along with the associated probability
        cvutils.draw(
            frame,
            # (startX, startY, endX, endY),
            current_detection[0:4].astype("int"),
            "{:.2f}%".format(detection_confidence * 100),
        )

    print('\nEND FRAME\n')
    return frame
