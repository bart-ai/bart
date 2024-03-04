import cv2

# Draw a rectangle on the frame, with optional text
def draw(frame, rectangle, text=None, color=(0, 255, 0), thickness=2):
    (startX, startY, endX, endY) = rectangle
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)

    if text:
        ytext = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(
            frame,
            text,
            (startX, ytext),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
        )

# Blur a rectangle on the frame
def blur(
    frame,
    coords,
    border=True,
    kernelsize=3,
):
    # TODO: remove this failsafe!
    if any(coord <= 0 for coord in coords):
        return
    if border:
        draw(frame, coords, color=(0, 0, 0), thickness=1)
    (startX, startY, endX, endY) = coords
    roi = frame[startY:endY, startX:endX]
    # TODO: investigate cv2.bilateralFilter
    blurred_roi = cv2.GaussianBlur(roi, (kernelsize, kernelsize), 0)
    frame[startY:endY, startX:endX] = blurred_roi