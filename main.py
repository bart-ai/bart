from roboflow import Roboflow
import cv2
import sys
from tqdm import tqdm
import imutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip-frames", default=10, type=int)
parser.add_argument("--secs-refetch", default=1, type=int)
parser.add_argument("--confidence", default=0, type=int)
parser.add_argument("--overlap", default=0, type=int)
parser.add_argument("video")
args = parser.parse_args()

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

video = cv2.VideoCapture(args.video)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
outvid = cv2.VideoWriter(f'out{args.video}', fourcc, fps, (width, height))
numberframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

for i in tqdm(range(numberframes)):
    if (args.skip_frames and i % args.skip_frames == 0):
        continue
    ret, frame = video.read()
    if not ret:
        break
    if frame is None:
        break
    frame = cv2.resize(frame, (width, height))

    if i % (args.secs_refetch * int(fps)) == 1:
        results = model.predict(
            frame,
            confidence=args.confidence,
            overlap=args.overlap,
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
            # Como escribimos la confidence aca?
    outvid.write(frame)

video.release()
outvid.release()
cv2.destroyAllWindows()
