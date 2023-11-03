import cv2


def overlap(box1, box2):
    (startX1, startY1, width1, height1) = box1
    (startX2, startY2, width2, height2) = box2
    x = max(startX1, startX2)
    y = max(startY1, startY2)
    w = min(startX1 + width1, startX2 + width2) - x
    h = min(startY1 + height1, startY2 + height2) - y
    return w > 0 or h > 0


def trackerToBbox(startX, startY, width, height):
    return (int(startX), int(startY), int(startX + width), int(startY + height))


def bboxToTracker(startX, startY, endX, endY):
    return (int(startX), int(startY), int(endX - startX), int(endY - startY))


def draw(frame, rectangle, text=None, color=(0, 255, 0)):
    (startX, startY, endX, endY) = rectangle
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if text:
        ytext = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(
            frame,
            text,
            (startX, ytext),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
        )
