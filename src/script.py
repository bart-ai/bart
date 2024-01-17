import argparse
import os

import core
import cv2
from cvutils import draw, trackerToBbox
from imutils.video import FPS
from tqdm import trange

# Command line parsing
parser = argparse.ArgumentParser(
    prog="main.py",
    description="Video object detector",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--target-fps",
    default=15,
    type=int,
    help="Frames per second of the output video\nLower values result in better performance\n(default: %(default)s)",
)
parser.add_argument(
    "--secs-refetch",
    default=0,
    type=int,
    help="Number of seconds before refetching the model\nObjects detected by the model are tracked between refetches\n(example: 0 == there's no tracking involved)\n(example: 2 == the model is refetched every 2 seconds)\n(default: %(default)s)",
)
parser.add_argument(
    "--confidence",
    default=0.8,
    type=int,
    help="Confidence threshold to use when detecting objects\n(example: 0.15 == 15%% confidence)\n(default: %(default)s)",
)
# TODO: make output name optional
parser.add_argument("video", help="video file to process")
args = parser.parse_args()

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

# Save output in the same directory as the input one, with the `out` prefix
videodirectory = os.path.dirname(args.video)
videobasename = os.path.basename(args.video)
outvideoname = os.path.join(videodirectory, f"out-{videobasename}")

# Set the output video to the input video dimensions and the desired fps
# To make it web-friendly, convert it with:
# `ffmpeg -i <input> -vcodec libx264 <output>`
sourcefps = round(video.get(cv2.CAP_PROP_FPS))
targetfps = min(args.target_fps, sourcefps)
skipframes = round(sourcefps / targetfps) if targetfps != sourcefps else 0
targetframes = round(numberframes / skipframes) if skipframes else numberframes
# TODO: Make the neccessary setup to be able to use the h264 codec here and
# make it parametrizeable
# TODO: Play and parametrize with VIDEOWRITER_PROP_QUALITY
outvid = cv2.VideoWriter(
    outvideoname,
    cv2.VideoWriter_fourcc(*"mp4v"),
    targetfps,
    (width, height),
)

# Loop over the frames
timer = FPS().start()
with trange(targetframes) as pbar:
    for framenumber in pbar:
        if args.secs_refetch and framenumber % (args.secs_refetch * targetfps) != 0:
            framestatus = "track"
        else:
            framestatus = "model"
        pbar.set_postfix(framestatus=framestatus)

        ret, frame = video.read()

        # Failsafe: we shouldn't enter here
        if not ret or frame is None:
            break

        # FRAME: run the model
        if framestatus == "model":
            core.detect(frame, args.confidence)
        # TODO: Recover this code once we start using the tracking functionality
        # # check if we are already tracking this object by a simple overlap
        # trackingBox = bboxToTracker(startX, startY, endX, endY)
        # tracked = False
        # for trackedBoxes in trackers.getObjects():
        #     if overlap(trackingBox, trackedBoxes):
        #         tracked = True
        #         break

        # # if we were not tracking the object, add it to our multitracker
        # if not tracked:
        #     tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
        #     trackers.add(tracker, frame, trackingBox)

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

        # Skip the next frames in order to keep the target fps
        # When we need to skip lots of frames (hardcoded at 100), it's better to
        # use the `set` method instead of reading the frames one by one
        if skipframes >= 100:
            video.set(cv2.CAP_PROP_POS_FRAMES, framenumber * skipframes)
        else:
            for _ in range(skipframes - 1):
                video.read()

        timer.update()
timer.stop()

# Release the opened output video to then be able to read it again
outvid.release()


# Open up the output video again to get its specs
outvid = cv2.VideoCapture(outvideoname)
outvidframes = int(outvid.get(cv2.CAP_PROP_FRAME_COUNT))
outvidfps = outvid.get(cv2.CAP_PROP_FPS)

print(
    f"""Summary: processed {round(duration, 2)} seconds of video in {round(timer.elapsed(), 2)} seconds ({round(timer.fps(), 2)} fps)
    Input {args.video}: {numberframes} frames  @ {fps} fps
    Output {outvideoname}: {outvidframes} frames  @ {outvidfps} fps
    """
)

# Release the video files
video.release()
outvid.release()
