import argparse
import time
import os

import cv2
import numpy as np
from tqdm import tqdm

from cvutils import bboxToTracker, draw, overlap, trackerToBbox

# Command line parsing
parser = argparse.ArgumentParser(
    prog="main.py",
    description="Video object detector",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--skip-frames",
    default=3,
    type=int,
    help="Number of frames to skip on the output video, helps performance\n0 == no frames are skipped; 10 == 1 out of 10 frames is skipped\n(default: 3)",
)
parser.add_argument(
    "--frames-refetch",
    default=0,
    type=int,
    help="Number of frames before refetching the model, ¿helps performance?\n0 == the model is refetched every frame; 10 == the model is refetched every 10 frames\n(default: 0)",
)
parser.add_argument(
    "--confidence",
    default=0.8,
    type=int,
    help="Confidence threshold to use when detecting objects\n0.8 == 80%% confidence\n(default: 0.8)",
)
parser.add_argument("video", help="mp4 video file to process")
args = parser.parse_args()

# Set up the model
net = cv2.dnn.readNetFromCaffe(
    "model/deploy.prototxt", "model/res10_300x300_ssd_iter_140000.caffemodel"
)

# opencv's built-in object trackers
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create,
}

# Keep track of multiple objects at the same time
trackers = cv2.legacy.MultiTracker_create()

# Input video specs
# TODO: make video arg optional, and if not present, record from webcam
# (and save output file with today's date)
video = cv2.VideoCapture(args.video)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
numberframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = numberframes / fps

videodirectory = os.path.dirname(args.video)
videobasename = os.path.basename(args.video)
outvideoname = os.path.join(videodirectory, f"out-{videobasename}")

# Save the output video with the same specs as the input one
# TODO: make it work for non mp4 files, ¿investigate other codecs?
outvid = cv2.VideoWriter(
    outvideoname,
    cv2.VideoWriter_fourcc(*"mp4v"),
    video.get(cv2.CAP_PROP_FPS),
    (width, height),
)


# Loop over the frames
start = time.time()
for framenumber in tqdm(range(numberframes)):
    # calling the model / skipping the frame / calling our tracker logic:
    # - The very first frame always calls the model
    # - Every skip-frames, I should skip the frame
    # - Every frames-refetch, I should refetch the model, if not, I should track the frame
    # - We shouldn't skip the frame if we are going to refetch the model
    if framenumber == 0 or (
        args.frames_refetch and framenumber % args.frames_refetch == 0
    ):
        framestatus = "model"
    elif args.skip_frames and framenumber % args.skip_frames == 0:
        framestatus = "skip"
    elif args.frames_refetch and framenumber % args.frames_refetch != 0:
        framestatus = "track"
    else:
        framestatus = "model"

    # FRAME: skip it
    if framestatus == "skip":
        continue

    ret, frame = video.read()

    # Failsafe: we shouldn't enter here
    if not ret or frame is None:
        break

    # FRAME: run the model
    if framestatus == "model":
        # Feed the frame to the model
        # TODO: the 300x300 comes from the example model, can we un-hardcode it?
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
        net.setInput(blob)

        # Run the model itself
        detections = net.forward()

        # mostly taken from https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if not confidence > args.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            # multiply by the frame size to get the actual coordinates
            (startX, startY, endX, endY) = (
                detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            ).astype("int")

            # draw the bounding box along with the associated probability
            draw(
                frame,
                (startX, startY, endX, endY),
                "{:.2f}%".format(confidence * 100),
            )

            # check if we are already tracking this object by a simple overlap
            trackingBox = bboxToTracker(startX, startY, endX, endY)
            tracked = False
            for trackedBoxes in trackers.getObjects():
                if overlap(trackingBox, trackedBoxes):
                    tracked = True
                    break

            # if we were not tracking the object, add it to our multitracker
            if not tracked:
                tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
                trackers.add(tracker, frame, trackingBox)

    # FRAME: don't run the model, use our multitracker
    elif framestatus == "track":
        # Multitracking brings a whole set of problems to solve, but right now
        # it doesn't appear to give any benefits. When we decide to activate it
        # we should check
        # - Instead of having a fixed number of frames before refetching the
        # model, there might be a dynamic way to decide when to refetch it.
        # maybe with a phash we can check if the image has changed enough from
        # our last model call? maybe opencv has an img diff function?
        # - What happens if the object goes out of frame? should we remove it
        # from our multitracker? And what happens when it re-enters the frame?
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (startX, startY, endX, endY) = trackerToBbox(*box)
            draw(frame, (startX, startY, endX, endY), color=(0, 255, 255))

    # Write the frame to the output video
    outvid.write(frame)
end = time.time()

print(
    f"Summary: processed {round(duration, 2)} seconds of video ({numberframes} frames) in {round(end-start, 2)} seconds"
)

# Release the video files
video.release()
outvid.release()
