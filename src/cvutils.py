import cv2


# Check if there's any overlap between two boxes
# TODO: add threshold parameter
def overlap(box1, box2):
    (startX1, startY1, width1, height1) = box1
    (startX2, startY2, width2, height2) = box2
    x = max(startX1, startX2)
    y = max(startY1, startY2)
    w = min(startX1 + width1, startX2 + width2) - x
    h = min(startY1 + height1, startY2 + height2) - y
    return w > 0 or h > 0


# Convert a (startx, starty, width, height) tuple to a (startx, starty, endx, endy) tuple
def trackerToBbox(startX, startY, width, height):
    return (int(startX), int(startY), int(startX + width), int(startY + height))


# Convert a (startx, starty, endx, endy) tuple to a (startx, starty, width, height) tuple
def bboxToTracker(startX, startY, endX, endY):
    return (int(startX), int(startY), int(endX - startX), int(endY - startY))


# Draw a rectangle on the frame, with optional text
def draw(frame, rectangle, text=None, color=(0, 255, 0)):
    (startX, startY, endX, endY) = rectangle
    print('drawing rectangle', startX, startY, endX, endY)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    print('done drawing rectangle')

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
