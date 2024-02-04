import os

import cv2
import cvutils
import numpy as np


def get_model():
    cwd = os.path.dirname(os.path.realpath(__file__))
    net = cv2.dnn.readNet(
        f"{cwd}/model/face-detection/res10_300x300_ssd_iter_140000.caffemodel",
        f"{cwd}/model/face-detection/deploy.prototxt",
    )
    dimensions = (300, 300)
    model = (net, dimensions)
    return model


def detect(frame, confidence=0.8, model=get_model()):
    # Feed the frame to the model
    net, net_dimensions = model
    blob = cv2.dnn.blobFromImage(frame, 1.0, net_dimensions)
    net.setInput(blob)

    # Run the model itself
    detections = net.forward()

    # mostly taken from https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the prediction
        detection_confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if not detection_confidence > confidence:
            continue

        # compute the (x, y)-coordinates of the bounding box for the object
        # multiply by the frame size to get the actual coordinates
        height, width = frame.shape[:2]
        (startX, startY, endX, endY) = (
            detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        ).astype("int")

        # draw the bounding box along with the associated probability
        cvutils.draw(
            frame,
            (startX, startY, endX, endY),
            "{:.2f}%".format(detection_confidence * 100),
        )

    return frame
