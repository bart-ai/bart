def calculate_total_area_covered_by_bboxes(bounding_boxes):
    total_area = 0
    visited = set()  # Keep track of visited bounding boxes to avoid double counting overlap

    for bbox in bounding_boxes:
        area = (bbox["endX"] - bbox["startX"]) * (bbox["endY"] - bbox["startY"])
        total_area += area

        # Check for overlap with other bounding boxes
        for other_bbox in bounding_boxes:
            if other_bbox == bbox or other_bbox in visited:
                continue

            # Calculate overlap area
            overlap_startX = max(bbox["startX"], other_bbox["startX"])
            overlap_endX = min(bbox["endX"], other_bbox["endX"])
            overlap_startY = max(bbox["startY"], other_bbox["startY"])
            overlap_endY = min(bbox["endY"], other_bbox["endY"])

            if overlap_startX < overlap_endX and overlap_startY < overlap_endY:
                overlap_area = (overlap_endX - overlap_startX) * (overlap_endY - overlap_startY)
                total_area -= overlap_area
                visited.add(other_bbox)
    
    return total_area
