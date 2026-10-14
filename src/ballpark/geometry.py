"""Field geometry. Coordinates: feet from home plate, +y toward center field,
+x toward right field. Spray angle 0 = dead center, negative = left field."""
import json

import numpy as np
from matplotlib.path import Path

from .paths import SEATMAPS

ANGLES = np.array([-45, -30, -15, 0, 15, 30, 45])
DIM_KEYS = ["leftLine", "left", "leftCenter", "center", "rightCenter", "right", "rightLine"]


def spray(hc_x, hc_y):
    """Statcast hit-coordinate -> feet, angle (deg), distance (ft)."""
    x = 2.5 * (np.asarray(hc_x, float) - 125.42)
    y = 2.5 * (198.27 - np.asarray(hc_y, float))
    return x, y, np.degrees(np.arctan2(x, y)), np.hypot(x, y)


def to_xy(angle, dist):
    r = np.radians(angle)
    return dist * np.sin(r), dist * np.cos(r)


def fence_distance(angle, park) -> np.ndarray:
    """Wall distance at a spray angle, linearly interpolated between the 7
    published points (assumed evenly spaced at 15 deg). Foul territory clamps to the line."""
    prof = [park[k] for k in DIM_KEYS]
    return np.interp(angle, ANGLES, prof)


def load_seatmap(venue_id: int):
    """Optional data/seatmaps/<venue_id>.json: [{"section": "...", "poly": [[x,y],...]}]
    with coordinates in feet from home plate."""
    f = SEATMAPS / f"{venue_id}.json"
    if not f.exists():
        return None
    return [(s["section"], Path(np.array(s["poly"]))) for s in json.loads(f.read_text())]


def seat_sections(x, y, seatmap):
    pts = np.column_stack([x, y])
    labels = np.full(len(pts), None, dtype=object)
    for name, path in seatmap:
        labels[path.contains_points(pts) & (labels == None)] = name  # noqa: E711
    return labels
