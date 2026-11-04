"""Unit tests for the stale-link hold helpers in scripts/multi_robot_explorer.py.

Run: python3 -m pytest <this file>  (needs rclpy importable, e.g. ROS 2 Humble sourced)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from multi_robot_explorer import (  # noqa: E402
    link_is_stale, link_resume_hold_active, link_retreat_blocks_assignment,
    stale_link_blocks_assignment, tf_freshness_update,
)


def test_fresh_time_moves_only_when_stamp_advances():
    assert tf_freshness_update(10.0, 100.0, 10.5, 101.0) == (10.5, 101.0)
    # Same transform returned again by a "latest" lookup: not fresh.
    assert tf_freshness_update(10.5, 101.0, 10.5, 105.0) == (10.5, 101.0)
    # Older stamp (out of order): ignored.
    assert tf_freshness_update(10.5, 101.0, 10.2, 105.0) == (10.5, 101.0)


def test_clock_offset_does_not_matter():
    # Robot clock 1000 s behind central: only stamp changes count.
    stamp, fresh = tf_freshness_update(0.0, 0.0, 50.0, 1050.0)
    stamp, fresh = tf_freshness_update(stamp, fresh, 50.1, 1050.1)
    assert fresh == 1050.1
    assert not link_is_stale(True, 1052.0, fresh, 3.0)


def test_static_only_chain_never_fresh_never_stale():
    stamp, fresh = tf_freshness_update(0.0, 0.0, 0.0, 100.0)
    assert fresh == 0.0
    assert not link_is_stale(True, 200.0, fresh, 3.0)


def test_stale_threshold():
    assert not link_is_stale(True, 102.9, 100.0, 3.0)
    assert link_is_stale(True, 103.1, 100.0, 3.0)


def test_disabled_never_stale_or_held():
    assert not link_is_stale(False, 1000.0, 100.0, 3.0)
    assert not link_resume_hold_active(False, 100.5, 100.0, 5.0)
    assert not stale_link_blocks_assignment(False, True, True)


def test_never_received_is_not_stale():
    assert not link_is_stale(True, 1000.0, 0.0, 3.0)


def test_resume_hold_window():
    assert link_resume_hold_active(True, 104.9, 100.0, 5.0)
    assert not link_resume_hold_active(True, 105.1, 100.0, 5.0)
    assert not link_resume_hold_active(True, 104.9, 0.0, 5.0)
    assert not link_resume_hold_active(True, 100.1, 100.0, 0.0)


def test_assignment_blocked_by_either_hold():
    # (pause_on_link_retreat active, stale hold active) -> blocked
    for retreat, stale, blocked in [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]:
        got = (
            link_retreat_blocks_assignment(True, retreat)
            or stale_link_blocks_assignment(True, stale, False)
        )
        assert got == blocked


def _replay(rows, tick_sec, after_sec, holdoff_sec):
    """Feed recorded TF through the helpers at explorer tick rate.

    Returns (first stale tick, first fresh-again tick, ticks that would have
    allowed a new request).
    """
    stamp = fresh = 0.0
    stale_active = False
    fresh_again = 0.0
    first_stale = first_fresh_again = None
    allowed = []
    i = 0
    t = rows[0][0]
    end = rows[-1][0]
    while t <= end:
        while i < len(rows) and rows[i][0] <= t:
            stamp, fresh = tf_freshness_update(stamp, fresh, rows[i][1], rows[i][0])
            i += 1
        stale = link_is_stale(True, t, fresh, after_sec)
        if stale and not stale_active:
            stale_active = True
            first_stale = first_stale or t
        elif not stale and stale_active:
            stale_active = False
            fresh_again = t
            first_fresh_again = first_fresh_again or t
        hold = link_resume_hold_active(True, t, fresh_again, holdoff_sec)
        if not stale_link_blocks_assignment(True, stale_active, hold):
            allowed.append(t)
        t += tick_sec
    return first_stale, first_fresh_again, allowed


def test_recorded_zenoh_freeze_replay():
    with open(os.path.join(HERE, "data", "tf_freeze_replay.json")) as f:
        doc = json.load(f)
    rows = doc["rows"]
    freeze_start = doc["freeze_start"]
    first_stale, fresh_again, allowed = _replay(rows, 0.1, 3.0, 5.0)
    # Last TF before the freeze arrived at ~112.7 s: stale ~3 s later.
    assert 115.0 <= first_stale <= 116.5
    # Central TF resumed at ~140.9 s (8 s after the bridge was resumed).
    assert 140.5 <= fresh_again <= 141.5
    # No request may be sent from the stale point until 5 s after resume.
    # The recorded run sent its fatal precheck at ~136.4 s and its
    # stall-watchdog cancel at ~130.7 s; both fall inside the hold.
    blocked = [t for t in allowed if first_stale <= t < fresh_again + 5.0]
    assert blocked == []
    assert any(t >= fresh_again + 5.0 for t in allowed)
    assert all(first_stale <= x < fresh_again + 5.0 for x in (130.7, 136.4))
    # Before the freeze the robot was assignable.
    assert any(t < freeze_start for t in allowed)
