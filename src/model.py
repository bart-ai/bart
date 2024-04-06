import os

import cv2
import numpy as np

import cvutils

IMAGE_SIZE = 640

class Model:

    detect_faces = 'face'
    detect_billboards = 'billboard'

    def _load_onnx_model(self, model_name):
        cwd = os.path.dirname(os.path.realpath(__file__))
        net = cv2.dnn.readNet(
            # YOLOv8 ONNX Model
            f"{cwd}/model/billboard-detection/{model_name}.onnx",
        )
        dimensions = (IMAGE_SIZE, IMAGE_SIZE)
        model = (net, dimensions)
        return model

    def _load_caffe_model(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        net = cv2.dnn.readNet(
            # Caffe Model
            f"{cwd}/model/face-detection/res10_300x300_ssd_iter_140000.caffemodel",
            f"{cwd}/model/face-detection/deploy.prototxt",
        )
        dimensions = (IMAGE_SIZE, IMAGE_SIZE)
        model = (net, dimensions)
        return model

    def _detect_onnx_model(self, frame, transformation, confidence):
        # Feed the frame to the model
        net, dimensions = self.model

        # Read the input image
        original_image = frame
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / IMAGE_SIZE

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
            # TODO: investigate why some of this coordinates might be negative
            startX = round(box[0] * scale)
            startY = round(box[1] * scale)
            endX = round((box[0] + box[2]) * scale)
            endY = round((box[1] + box[3]) * scale)
            self.transform(frame, (startX, startY, endX, endY), scores[index], transformation)

        return frame

    def _detect_caffe_model(self, frame, transformation, confidence):
        # Feed the frame to the model
        net, net_dimensions = self.model
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
            self.transform(
                frame,
                (startX, startY, endX, endY),
                detection_confidence,
                transformation,
            )

        return frame

    def transform(self, frame, rectangle, score, transformation):
        if transformation == "detect":
            cvutils.draw(
                frame,
                rectangle,
                f"{self.object}: {score*100:.2f}%",
            )
        elif transformation == "blur":
            cvutils.blur(frame, rectangle)


    def __init__(self, model_type="billboards", model_name="yolov8n-e50"):
        self.model = self._load_onnx_model(model_name) if model_type == Model.detect_billboards else self._load_caffe_model()
        self.detect = self._detect_onnx_model if model_type == Model.detect_billboards else self._detect_caffe_model
        self.object = model_type

    def detect(self, frame, transformation = "detect", confidence = 0.8):
        self.detect(frame, transformation, confidence)
