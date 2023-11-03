import argparse

import cv2
import numpy as np
from tqdm import tqdm

from cvutils import bboxToTracker, draw, overlap, trackerToBbox

parser = argparse.ArgumentParser()
parser.add_argument("--skip-frames", default=10, type=int)
parser.add_argument("--secs-refetch", default=1, type=int)
parser.add_argument("--confidence", default=0.8, type=int)
parser.add_argument("video")
args = parser.parse_args()

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create,
}

trackers = cv2.legacy.MultiTracker_create()

video = cv2.VideoCapture(args.video)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

outvid = cv2.VideoWriter(
    f"out-{args.video}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

numberframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
trackers = cv2.legacy.MultiTracker_create()

for framenumber in tqdm(range(numberframes)):
    # Intentar que los skip frames no sean fijos, sean basados en la latencia/poder del cliente
    if args.skip_frames and framenumber % args.skip_frames == 0:
        continue

    ret, frame = video.read()

    # Failsafe: we shouldn't enter here
    if not ret or frame is None:
        break

    # Encontrar alguna manera dinamica de decidir cuando rellamar al modelo (un estilo de "cambio un 20% de la imagen, es hora de llamar de nuevo", en vez de que sea yn numero de frames fijo
    if not args.secs_refetch or framenumber % (args.secs_refetch * int(fps)) == 1:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                (startX, startY, endX, endY) = (
                    detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                ).astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                draw(
                    frame,
                    (startX, startY, endX, endY),
                    "{:.2f}%".format(confidence * 100),
                )

                # Check if we are already tracking this box
                trackingBox = bboxToTracker(startX, startY, endX, endY)
                tracked = False
                for trackedBoxes in trackers.getObjects():
                    if overlap(trackingBox, trackedBoxes):
                        tracked = True
                        break

                if not tracked:
                    tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
                    trackers.add(tracker, frame, trackingBox)

    else:
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (startX, startY, endX, endY) = trackerToBbox(*box)
            draw(frame, (startX, startY, endX, endY), color=(0, 255, 255))

    outvid.write(frame)

video.release()
outvid.release()
cv2.destroyAllWindows()
