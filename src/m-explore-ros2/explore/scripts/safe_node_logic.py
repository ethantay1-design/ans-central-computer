#!/usr/bin/env python3
"""
Pure safe-node logic for the central computer (no ROS imports).

Turns the merged occupancy grid (/map, 5 cm cells) plus a robot's Wi-Fi
quality samples into a coarse risk grid (0.5 m cells, 0 = safe, 100 = worst,
-1 = unknown) and picks the nearest safe node the robot can reach.

Inputs use the nav_msgs/OccupancyGrid conventions: row-major data, row 0 at
origin_y, values -1 unknown, 0 free, up to 100 occupied. Grid rotation is
assumed to be zero (true for map_merge's /map).

A ROS node (phase 2) wraps this module; the numbers below are proposals.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

UNKNOWN = -1

# Coarse cell states
FREE = 0
BLOCKED = 1
UNEXPLORED = 2


@dataclass
class RiskParams:
    coarse_resolution: float = 0.5
    # A fine cell at or above this value counts as an obstacle.
    occupied_threshold: int = 65
    # A coarse cell with less known area than this is unexplored.
    min_known_fraction: float = 0.5
    # Obstacle risk falls from 100 next to an obstacle to 0 at this distance.
    clearance_m: float = 1.0
    # Cells at or below this combined risk can be safe nodes.
    safe_threshold: int = 30


@dataclass
class CoarseGrid:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    state: List[int]

    def index(self, cx: int, cy: int) -> int:
        return cy * self.width + cx

    def cell_of(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        cx = int(math.floor((x - self.origin_x) / self.resolution))
        cy = int(math.floor((y - self.origin_y) / self.resolution))
        if 0 <= cx < self.width and 0 <= cy < self.height:
            return cx, cy
        return None

    def center_of(self, cx: int, cy: int) -> Tuple[float, float]:
        return (
            self.origin_x + (cx + 0.5) * self.resolution,
            self.origin_y + (cy + 0.5) * self.resolution,
        )


def quality_to_risk(quality: float) -> int:
    """Wi-Fi quality 0..1 (1 = best) to connection risk 0..100."""
    q = min(1.0, max(0.0, float(quality)))
    return int(round((1.0 - q) * 100.0))


def coarsen(
    data: Sequence[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    params: RiskParams,
) -> CoarseGrid:
    """Group fine cells into coarse cells: BLOCKED, FREE or UNEXPLORED."""
    factor = max(1, int(round(params.coarse_resolution / resolution)))
    cw = (width + factor - 1) // factor
    ch = (height + factor - 1) // factor
    occupied = [0] * (cw * ch)
    known = [0] * (cw * ch)
    total = [0] * (cw * ch)
    for y in range(height):
        row = y * width
        cy = y // factor
        for x in range(width):
            ci = cy * cw + x // factor
            v = data[row + x]
            total[ci] += 1
            if v != UNKNOWN:
                known[ci] += 1
                if v >= params.occupied_threshold:
                    occupied[ci] += 1
    state = []
    for ci in range(cw * ch):
        if occupied[ci] > 0:
            state.append(BLOCKED)
        elif known[ci] < params.min_known_fraction * total[ci]:
            state.append(UNEXPLORED)
        else:
            state.append(FREE)
    return CoarseGrid(cw, ch, resolution * factor, origin_x, origin_y, state)


def obstacle_distance(grid: CoarseGrid, max_m: float) -> List[float]:
    """
    Distance in metres from each cell centre to the nearest BLOCKED cell,
    searched only out to max_m (beyond that: inf). The window keeps this
    fast on a large merged map.
    """
    reach = int(math.ceil(max_m / grid.resolution)) + 1
    offsets = sorted(
        ((dx, dy) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)),
        key=lambda o: o[0] ** 2 + o[1] ** 2,
    )
    out = []
    for i in range(len(grid.state)):
        cx, cy = i % grid.width, i // grid.width
        d = math.inf
        for dx, dy in offsets:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height \
                    and grid.state[ny * grid.width + nx] == BLOCKED:
                d = math.sqrt(dx * dx + dy * dy) * grid.resolution
                break
        out.append(d if d <= max_m + grid.resolution else math.inf)
    return out


@dataclass
class SignalMap:
    """Running mean of Wi-Fi quality per coarse cell."""

    grid: CoarseGrid
    _sum: Dict[int, float] = field(default_factory=dict)
    _count: Dict[int, int] = field(default_factory=dict)

    def add_sample(self, x: float, y: float, quality: float) -> bool:
        cell = self.grid.cell_of(x, y)
        if cell is None:
            return False
        i = self.grid.index(*cell)
        self._sum[i] = self._sum.get(i, 0.0) + quality
        self._count[i] = self._count.get(i, 0) + 1
        return True

    def quality(self, i: int) -> Optional[float]:
        if i not in self._count:
            return None
        return self._sum[i] / self._count[i]


def risk_grid(
    grid: CoarseGrid, signal: SignalMap, params: RiskParams
) -> List[int]:
    """
    Combined risk per coarse cell: the worse of connection risk and obstacle
    risk. BLOCKED = 100. Cells without a Wi-Fi sample are UNKNOWN, because
    connection risk there has not been measured.
    """
    # Half a cell extra, because risk uses the distance to the cell edge.
    dist = obstacle_distance(grid, params.clearance_m + grid.resolution / 2.0)
    out = []
    for i, s in enumerate(grid.state):
        if s == BLOCKED:
            out.append(100)
            continue
        q = signal.quality(i)
        if s == UNEXPLORED or q is None:
            out.append(UNKNOWN)
            continue
        # Distance is centre to centre; subtract half a cell for the edge.
        edge = max(0.0, dist[i] - grid.resolution / 2.0)
        obstacle = int(round(100.0 * max(0.0, 1.0 - edge / params.clearance_m)))
        out.append(max(quality_to_risk(q), obstacle))
    return out


def select_safe_node(
    grid: CoarseGrid,
    risk: Sequence[int],
    robot_xy: Tuple[float, float],
    params: RiskParams,
) -> Optional[Tuple[float, float]]:
    """
    Nearest cell (by steps through non-blocked cells, so not through walls)
    whose risk is known and at or below safe_threshold. Returns the cell
    centre in map coordinates, or None if no such cell is reachable.
    """
    start = grid.cell_of(*robot_xy)
    if start is None:
        return None
    seen = {start}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        r = risk[grid.index(cx, cy)]
        if r != UNKNOWN and r <= params.safe_threshold:
            return grid.center_of(cx, cy)
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            if (nx, ny) in seen:
                continue
            if grid.state[grid.index(nx, ny)] == BLOCKED:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return None
