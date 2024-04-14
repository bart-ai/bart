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
    kernelsize=15,
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

# Calculate the total area covered by a list of bounding boxes
# TODO: Check if the 'line sweep' algorithm performs better
# https://stackoverflow.com/questions/5880558/intersection-of-n-rectangles
def calculate_total_area_covered_by_bboxes(bounding_boxes):
    total_area = 0
    # Keep track of visited bounding boxes to avoid double counting overlap
    visited = set()

    for bbox in bounding_boxes:
        (startX, startY, endX, endY) = bbox
        area = (endX - startX) * (endY - startY)
        total_area += area

        # Check for overlap with other bounding boxes
        for other_bbox in bounding_boxes:
            if other_bbox == bbox or other_bbox in visited:
                continue

            (other_startX, other_startY, other_endX, other_endY) = other_bbox

            # Calculate overlap area
            overlap_startX = max(startX, other_startX)
            overlap_endX = min(endX, other_endX)
            overlap_startY = max(startY, other_startY)
            overlap_endY = min(endY, other_endY)

            # If there is any overlap, then we must remove it from the total area
            if overlap_startX < overlap_endX and overlap_startY < overlap_endY:
                overlap_area = (overlap_endX - overlap_startX) * (overlap_endY - overlap_startY)
                total_area -= overlap_area
                visited.add(other_bbox)

    return total_area
