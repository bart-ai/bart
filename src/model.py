import os

import cv2
import numpy as np

import cvutils
import onnxruntime as ort

cwd = os.path.dirname(os.path.realpath(__file__))
BILLBOARD_MODELS_DIR = f"{cwd}/model/billboard-detection"

TRANSFORMATIONS = ["detect", "blur", "inpaint"]
FACE_MODEL_IMAGE_SIZE = 300
BILLBOARD_MODEL_IMAGE_SIZE = 640

OBJECT = {"FACES": "face", "BILLBOARDS": "billboard"}
AVAILABLE_MODELS = [
    {
        "name": "Bart Main Model",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e200-multi-bestparams.onnx",
    },
    {
        "name": "s2oeb dataset - 8155 images",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e60-s2oeb-bestparams.onnx",
    },
    {
        "name": "ydtns dataset - 2719 images",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e60-ydtns-bestparams.onnx",
    },
    {
        "name": "isfau dataset - 7137 images",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e60-isfau-bestparams.onnx",
    },
    {
        "name": "uo2ld dataset - 1903 images",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e60-uo2ld-bestparams.onnx",
    },
    {
        "name": "hackathon dataset - 4503 images",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-e60-hackathon-bestparams.onnx",
    },
    {
        "name": "OpenImages v7 nano model",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8n-oiv7.onnx",
    },
    {
        "name": "OpenImages v7 small model",
        "detection": OBJECT["BILLBOARDS"],
        "filename": "yolov8s-oiv7.onnx",
    },
    {"name": "Native Face Detection", "detection": OBJECT["FACES"]},
]
# Add the rest of the models in the billboards directory
filenames = [model.get("filename") for model in AVAILABLE_MODELS if "filename" in model]
AVAILABLE_MODELS.extend(
    [
        {
            "name": model_filename.replace(".onnx", ""),
            "filename": model_filename,
            "detection": OBJECT["BILLBOARDS"],
        }
        for model_filename in os.listdir(BILLBOARD_MODELS_DIR)
        if model_filename not in filenames
    ]
)


class Model:
    def _set_onnx_model(self, model_filename):
        self.dimensions = (BILLBOARD_MODEL_IMAGE_SIZE, BILLBOARD_MODEL_IMAGE_SIZE)

        model_path = f"{cwd}/model/billboard-detection/{model_filename}"
        if self.ort:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.model = cv2.dnn.readNet(model_path)

    def _set_caffe_model(self):
        self.dimensions = (FACE_MODEL_IMAGE_SIZE, FACE_MODEL_IMAGE_SIZE)

        net = cv2.dnn.readNet(
            f"{cwd}/model/face-detection/res10_300x300_ssd_iter_140000.caffemodel",
            f"{cwd}/model/face-detection/deploy.prototxt",
        )
        self.model = net

    def _detect_onnx_model(self, frame, transformation, confidence):
        # Read the input image
        original_image = frame
        [height, width, _] = original_image.shape

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        # Keep track of the scaling factors for the bounding boxes
        img = cv2.resize(img, self.dimensions)
        x_factor = width / self.dimensions[1]
        y_factor = height / self.dimensions[0]

        # Run with ONNX Runtime
        if self.ort:
            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0

            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

            # Run inference using the preprocessed image data
            outputs = self.session.run(None, {'images': image_data})

        # Run with OpenCV
        else:
            # Preprocess the image and prepare blob for model
            blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=self.dimensions, swapRB=True)
            self.model.setInput(blob)

            # Perform inference
            outputs = self.model.forward()

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        results = []
        # Iterate through output to collect bounding boxes and confidence scores
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # Skip if the maximum score isn't above the confidence threshold
            if max_score < confidence:
                continue

            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # OPENIMAGES CLASSES INDEXES: https://docs.ultralytics.com/datasets/detect/open-images-v7/
            # 46 -> Billboard
            # 264 -> Human face
            if (self.is_openimages and class_id != 46):
                continue

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            box = [
                int((x - w / 2) * x_factor), # left
                int((y - h / 2) * y_factor), # top
                int(w * x_factor), # width
                int(h * y_factor) # height
            ]
            results.append((box, max_score))

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(
            [result[0] for result in results],
            [result[1] for result in results],
            confidence or 0.5,
            0.45,
            0.5
        )

        # Here we'll keep track of all the bounding boxes to eventually calculate
        # the total area covered by the boxes in the frame.
        bounding_boxes = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box, score = results[index]

            startX = round(box[0])
            startY = round(box[1])
            endX = round((box[0] + box[2]))
            endY = round((box[1] + box[3]))

            # Some coordinates might be out of bounds
            # We clamp them to the image dimensions
            startX = max(0, min(startX, width-1))
            startY = max(0, min(startY, height-1))
            endX = max(0, min(endX, width-1))
            endY = max(0, min(endY, height-1))

            # Skip if the bounding box is too small
            if (startX == endX or startY == endY):
                continue

            bbox = (startX, startY, endX, endY)
            self.transform(frame, bbox, score, transformation)
            bounding_boxes.append(bbox)

        detection_area = cvutils.calculate_total_area_covered_by_bboxes(bounding_boxes)
        detection_area_percentage = detection_area / (height * width) * 100

        return frame, detection_area_percentage

    def _detect_caffe_model(self, frame, transformation, confidence):
        # Feed the frame to the model
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.dimensions)
        self.model.setInput(blob)

        [height, width, _] = frame.shape

        # Run the model itself
        detections = self.model.forward()

        # Here we'll keep track of all the bounding boxes to eventually calculate
        # the total area covered by the boxes in the frame.
        bounding_boxes = []

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
            bbox = tuple((
                detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            ).astype("int"))

            self.transform(frame, bbox, detection_confidence, transformation)
            bounding_boxes.append(bbox)

        detection_area = cvutils.calculate_total_area_covered_by_bboxes(bounding_boxes)
        detection_area_percentage = detection_area / (height * width) * 100

        return frame, detection_area_percentage

    def transform(self, frame, rectangle, score, transformation):
        if transformation == "detect":
            cvutils.draw(
                frame,
                rectangle,
                f"{self.object}: {score*100:.2f}%",
            )
        elif transformation == "blur":
            cvutils.blur(frame, rectangle)
        elif transformation == "inpaint":
            cvutils.inpaint(frame, rectangle)

    def detect(self, frame, transformation = "detect", confidence = 0.8):
        frame, detection_area_percentage = self.detect(frame, transformation, confidence)
        return frame, detection_area_percentage

    def __init__(self, model, useOrt=True):
        self.ort = useOrt
        if model["detection"] == OBJECT["BILLBOARDS"]:
            self._set_onnx_model(model["filename"])
            self.detect = self._detect_onnx_model
        else:
            self._set_caffe_model()
            self.detect = self._detect_caffe_model
        self.is_openimages = "oiv7" in model.get("filename", "")
        self.object = model["detection"]
