"""Known facade-panel localization for the fixed demo building."""

from __future__ import annotations

from .schemas import FacadeLocation

COLUMNS = ("A", "B", "C", "D")
ROWS = ("1", "2", "3")


def locate_bbox(bbox_xyxy: list[float], image_width: int, image_height: int) -> FacadeLocation:
    x1, y1, x2, y2 = bbox_xyxy
    centroid_x = max(0.0, min(1.0, ((x1 + x2) / 2.0) / image_width))
    centroid_y = max(0.0, min(1.0, ((y1 + y2) / 2.0) / image_height))
    column_index = min(len(COLUMNS) - 1, int(centroid_x * len(COLUMNS)))
    row_index = min(len(ROWS) - 1, int(centroid_y * len(ROWS)))
    return FacadeLocation(
        panel_id=f"{COLUMNS[column_index]}{ROWS[row_index]}",
        normalized_centroid=[round(centroid_x, 5), round(centroid_y, 5)],
    )
