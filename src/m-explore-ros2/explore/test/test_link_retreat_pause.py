"""Unit tests for the link-retreat pause helpers in scripts/multi_robot_explorer.py.

Run: python3 -m pytest <this file>  (needs rclpy importable, e.g. ROS 2 Humble sourced)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from multi_robot_explorer import (  # noqa: E402
    cancel_blacklist_to_undo, link_retreat_blocks_assignment,
)


def test_blocks_only_when_enabled_and_active():
    assert link_retreat_blocks_assignment(True, True)
    assert not link_retreat_blocks_assignment(True, False)
    assert not link_retreat_blocks_assignment(False, True)
    assert not link_retreat_blocks_assignment(False, False)


def test_undo_recent_cancel_blacklist():
    assert cancel_blacklist_to_undo(100.5, 100.0, (1.0, 2.0), 3.0) == (1.0, 2.0)


def test_keep_old_cancel_blacklist():
    assert cancel_blacklist_to_undo(110.0, 100.0, (1.0, 2.0), 3.0) is None


def test_nothing_to_undo():
    assert cancel_blacklist_to_undo(100.5, 0.0, (1.0, 2.0), 3.0) is None
    assert cancel_blacklist_to_undo(100.5, 100.0, None, 3.0) is None


def test_cancel_after_flag_is_not_undone():
    # A cancel stamped after the retreat started is handled by the
    # skip-while-active check, not by this undo.
    assert cancel_blacklist_to_undo(100.0, 100.5, (1.0, 2.0), 3.0) is None
