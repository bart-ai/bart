from roboflow import Roboflow
import cv2
import sys
from tqdm import tqdm
import imutils

rf = Roboflow(api_key="aVZZ7kyDSgHYfCJt0DCr")
project = rf.workspace().project("billboards-cmzpu")
model = project.version(3).model


def objToBbox(obj):
    x1 = obj["x"] - (obj["width"] / 2)
    y1 = obj["y"] - (obj["height"] / 2)
    x2 = obj["x"] + (obj["width"] / 2)
    y2 = obj["y"] + (obj["height"] / 2)
    w = obj["width"]
    h = obj["height"]
    return x1, y1, x2, y2, w, h


def bboxToObject(bbox):
    x1, y1, w, h = bbox
    x1 = int(x1)
    y1 = int(y1)
    w = int(w)
    h = int(h)
    return x1, y1, x1 + w, y1 + h


def bboxToTracker(bbox):
    x1, y1, x2, y2, w, h = bbox
    return (x1, y1, w, h)


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

trackers = cv2.legacy.MultiTracker_create()

# model.predict(
#     "test.jpg",
#     confidence=0,
#     overlap=0,
# ).save("roboflow.jpg")

# img = cv2.imread("test.jpg")
video = cv2.VideoCapture("test.mkv")
fps = video.get(cv2.CAP_PROP_FPS)
width = int(640)
height = int(480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvid = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
numberframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

for i in tqdm(range(numberframes)):
    if (i != 48 and i % 10 == 0):
        continue
    ret, frame = video.read()
    if not ret:
        break
    if frame is None:
        break
    frame = cv2.resize(frame, (width, height))

    if i % 48 == 0:
        results = model.predict(
            frame,
            confidence=0,
            overlap=0,
        )
        trackers = cv2.legacy.MultiTracker_create()
        for bbox in [objToBbox(result) for result in results]:
            tracker = OPENCV_OBJECT_TRACKERS['kcf']()
            trackers.add(tracker, frame, bboxToTracker(bbox))
    else:
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            x1, y1, x2, y2 = bboxToObject(box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # for obj in results:
    #     x1 = int(x1)
    #     y1 = int(y1)
    #     x2 = int(x2)
    #     y2 = int(y2)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(img, f"Billboard {round(obj['confidence']*100, 2)}%",
    #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 2)

    outvid.write(frame)
    # cv2.imshow("image", frame)
    # if (cv2.waitKey(0)):
    #     break

video.release()
outvid.release()
cv2.destroyAllWindows()
