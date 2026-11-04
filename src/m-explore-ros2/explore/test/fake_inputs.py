"""
Fake inputs standing in for Blinky and map_merge (no ROS imports).

Builds a made-up floor plan as a 5 cm occupancy grid, a drive path through
it, and Wi-Fi quality along that path. The signal model is invented: it only
needs to be strong near the access point and weaker with distance and walls.
"""

import math
import random
from typing import List, Tuple

RESOLUTION = 0.05
ORIGIN = (0.0, 0.0)
WIDTH_M, HEIGHT_M = 12.0, 8.0

# Access point in room A (left).
ACCESS_POINT = (1.5, 6.5)

# Drive path waypoints: room A -> hallway door -> room B -> far corner.
WAYPOINTS = [
    (1.5, 1.5), (1.5, 6.0), (4.0, 6.0), (4.0, 2.0), (5.5, 2.0),
    (7.5, 2.0), (7.5, 6.5), (10.5, 6.5), (10.5, 1.5), (8.5, 1.5),
]


def _wall_cells(w: int, h: int) -> List[Tuple[int, int]]:
    cells = []
    t = 3  # wall thickness in cells (15 cm)
    for x in range(w):
        for k in range(t):
            cells += [(x, k), (x, h - 1 - k)]
    for y in range(h):
        for k in range(t):
            cells += [(k, y), (w - 1 - k, y)]
    # Inner wall at x = 6 m with a 1 m doorway at y = 1.5..2.5 m.
    wx = int(6.0 / RESOLUTION)
    for y in range(h):
        if 1.5 <= y * RESOLUTION < 2.5:
            continue
        for k in range(t):
            cells.append((wx + k, y))
    # A box (furniture) in room B.
    for x in range(int(8.5 / RESOLUTION), int(9.5 / RESOLUTION)):
        for y in range(int(3.5 / RESOLUTION), int(4.5 / RESOLUTION)):
            cells.append((x, y))
    return cells


def make_map() -> Tuple[List[int], int, int]:
    """Occupancy grid data (row-major), width, height. Top-right unexplored."""
    w, h = int(WIDTH_M / RESOLUTION), int(HEIGHT_M / RESOLUTION)
    data = [0] * (w * h)
    for x, y in _wall_cells(w, h):
        data[y * w + x] = 100
    # Unexplored corner of room B.
    for y in range(int(5.5 / RESOLUTION), h):
        for x in range(int(11.0 / RESOLUTION), w):
            data[y * w + x] = -1
    return data, w, h


def _walls_between(data, w, a, b) -> int:
    """Number of separate wall crossings on the straight line a -> b."""
    steps = int(math.dist(a, b) / RESOLUTION) + 1
    crossings, inside = 0, False
    for i in range(steps + 1):
        t = i / max(1, steps)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        v = data[int(y / RESOLUTION) * w + int(x / RESOLUTION)]
        if v >= 65 and not inside:
            crossings += 1
        inside = v >= 65
    return crossings


def wifi_dbm(data, w, xy, rng: random.Random) -> float:
    """Invented path-loss model: -35 dBm at 1 m, -20 dB/decade, -10 dB/wall."""
    d = max(0.5, math.dist(xy, ACCESS_POINT))
    walls = _walls_between(data, w, xy, ACCESS_POINT)
    return -35.0 - 20.0 * math.log10(d) * 1.6 - 10.0 * walls + rng.gauss(0, 2)


def dbm_to_quality(dbm: float, good: float = -50.0, bad: float = -80.0) -> float:
    """Same proposal as the robot side: -50 dBm or better = 1, -80 or worse = 0."""
    return min(1.0, max(0.0, (dbm - bad) / (good - bad)))


def make_path(spacing: float = 0.25) -> List[Tuple[float, float]]:
    path = []
    for a, b in zip(WAYPOINTS, WAYPOINTS[1:]):
        n = max(1, int(math.dist(a, b) / spacing))
        for i in range(n):
            t = i / n
            path.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    path.append(WAYPOINTS[-1])
    return path


def make_samples(seed: int = 0):
    """(map data, width, height, [(x, y, quality), ...]) along the path."""
    data, w, h = make_map()
    rng = random.Random(seed)
    samples = [
        (x, y, dbm_to_quality(wifi_dbm(data, w, (x, y), rng)))
        for x, y in make_path()
    ]
    return data, w, h, samples
