"""Unit tests for scripts/safe_node_logic.py. Run: python3 -m pytest <this file>"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)

import fake_inputs  # noqa: E402
from safe_node_logic import (  # noqa: E402
    BLOCKED, FREE, UNEXPLORED, UNKNOWN, RiskParams, SignalMap, coarsen,
    quality_to_risk, risk_grid, select_safe_node,
)

PARAMS = RiskParams()


def build(samples_upto=None):
    data, w, h, samples = fake_inputs.make_samples()
    grid = coarsen(data, w, h, fake_inputs.RESOLUTION, 0.0, 0.0, PARAMS)
    signal = SignalMap(grid)
    for x, y, q in samples[:samples_upto]:
        signal.add_sample(x, y, q)
    return grid, signal, risk_grid(grid, signal, PARAMS), samples


def test_quality_to_risk_bounds():
    assert quality_to_risk(1.0) == 0
    assert quality_to_risk(0.0) == 100
    assert quality_to_risk(0.5) == 50
    assert quality_to_risk(2.0) == 0
    assert quality_to_risk(-1.0) == 100


def test_coarsen_size_and_states():
    grid, _, _, _ = build()
    assert (grid.width, grid.height) == (24, 16)
    assert grid.resolution == 0.5
    # Outer wall, open floor in room A, unexplored corner of room B.
    assert grid.state[grid.index(0, 0)] == BLOCKED
    assert grid.state[grid.index(*grid.cell_of(2.0, 4.0))] == FREE
    assert grid.state[grid.index(*grid.cell_of(11.9, 3.0))] == BLOCKED  # wall
    assert grid.state[grid.index(*grid.cell_of(11.2, 6.2))] == UNEXPLORED


def test_risk_values_in_range_and_walls_max():
    grid, _, risk, _ = build()
    assert len(risk) == grid.width * grid.height
    for i, r in enumerate(risk):
        assert r == UNKNOWN or 0 <= r <= 100
        if grid.state[i] == BLOCKED:
            assert r == 100


def test_unvisited_cells_unknown():
    grid, _, risk, _ = build()
    # Room A's bottom-right area is never driven through.
    assert risk[grid.index(*grid.cell_of(5.0, 4.5))] == UNKNOWN


def test_safe_node_is_free_safe_and_visited():
    grid, signal, risk, samples = build()
    for x, y, _ in samples[::5]:
        node = select_safe_node(grid, risk, (x, y), PARAMS)
        if node is None:
            continue
        i = grid.index(*grid.cell_of(*node))
        assert grid.state[i] == FREE
        assert signal.quality(i) is not None
        assert risk[i] <= PARAMS.safe_threshold


def test_safe_node_found_near_access_point():
    grid, _, risk, _ = build()
    node = select_safe_node(grid, risk, (1.5, 5.5), PARAMS)
    assert node is not None
    # Within 1 m of the robot: room A has strong signal.
    assert abs(node[0] - 1.5) <= 1.0 and abs(node[1] - 5.5) <= 1.0


def test_safe_node_from_dead_zone_is_in_room_a_via_door():
    # Far corner of room B has weak signal behind the inner wall.
    grid, _, risk, _ = build()
    far = (10.5, 6.5)
    assert risk[grid.index(*grid.cell_of(*far))] > PARAMS.safe_threshold
    node = select_safe_node(grid, risk, far, PARAMS)
    assert node is not None
    # All of room B is risky, so the node is back through the door in room A.
    assert node[0] < 6.0
    assert 1.5 <= node[1] <= 2.5


def test_no_samples_means_no_safe_node():
    grid, _, risk, _ = build(samples_upto=0)
    assert all(r in (UNKNOWN, 100) for r in risk)
    assert select_safe_node(grid, risk, (2.0, 2.0), PARAMS) is None


def test_robot_outside_map_returns_none():
    grid, _, risk, _ = build()
    assert select_safe_node(grid, risk, (-5.0, -5.0), PARAMS) is None


def test_path_does_not_cross_walls():
    # Search must not jump the inner wall: block the doorway and check that
    # a robot in room B gets nothing from room A.
    data, w, h, samples = fake_inputs.make_samples()
    res = fake_inputs.RESOLUTION
    for y in range(int(1.5 / res), int(2.5 / res)):
        for x in range(int(6.0 / res), int(6.0 / res) + 3):
            data[y * w + x] = 100
    grid = coarsen(data, w, h, res, 0.0, 0.0, PARAMS)
    signal = SignalMap(grid)
    for x, y, q in samples:
        if x < 6.0:
            signal.add_sample(x, y, q)
    risk = risk_grid(grid, signal, PARAMS)
    assert select_safe_node(grid, risk, (8.0, 3.0), PARAMS) is None


def test_obstacle_distance_capped_and_fast():
    import time
    from safe_node_logic import obstacle_distance
    # 50 m x 50 m merged map (1000 x 1000 fine cells) with walls every 5 m.
    w = h = 1000
    data = [0] * (w * h)
    for y in range(0, h, 100):
        for x in range(w):
            data[y * w + x] = 100
    t0 = time.monotonic()
    grid = coarsen(data, w, h, 0.05, 0.0, 0.0, PARAMS)
    risk = risk_grid(grid, SignalMap(grid), PARAMS)
    elapsed = time.monotonic() - t0
    assert len(risk) == 100 * 100
    assert elapsed < 2.0, f"risk grid took {elapsed:.2f} s"
    dist = obstacle_distance(grid, 1.0)
    # Cell 2.5 m from the nearest wall is beyond the cap.
    assert dist[grid.index(10, 5)] == float("inf")
    assert dist[grid.index(10, 1)] == 0.5
