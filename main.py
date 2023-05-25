from roboflow import Roboflow
import cv2
import sys
from tqdm import tqdm

rf = Roboflow(api_key="roboflowapikey")
project = rf.workspace().project("billboards-cmzpu")
model = project.version(3).model


def objtobb(obj):
    x1 = obj["x"] - (obj["width"] / 2)
    y1 = obj["y"] - (obj["height"] / 2)
    x2 = obj["x"] + (obj["width"] / 2)
    y2 = obj["y"] + (obj["height"] / 2)
    w = obj["width"]
    h = obj["height"]
    return x1, y1, x2, y2, w, h


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    # "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    # "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# model.predict(
#     "test.jpg",
#     confidence=0,
#     overlap=0,
# ).save("roboflow.jpg")

# img = cv2.imread("test.jpg")
video = cv2.VideoCapture("test.mkv")
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvid = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

bbox = None

for i in tqdm(range(129)):
    ret, frame = video.read()
    if not ret:
        break
    if frame is None:
        break

    if (bbox is None):
        results = model.predict(
            frame,
            confidence=0,
            overlap=0,
        )
        bbox = objtobb(results[1])
        tracker.init(frame,
                     (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
    else:
        success, bbox = tracker.update(frame)
        if success:
            x1, y1, w, h = bbox
            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            bbox = None

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
