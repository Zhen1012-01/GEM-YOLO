from itertools import pairwise

import pytest
from gem_lane_yolo.control import (
    PIDController,
    clamp,
    estimate_center_from_single,
    select_lane_pair,
)


def test_clamp_and_invalid_limits():
    assert clamp(3.0, -1.0, 1.0) == 1.0
    assert clamp(-3.0, -1.0, 1.0) == -1.0
    with pytest.raises(ValueError):
        clamp(0.0, 1.0, -1.0)


def test_pid_respects_slew_and_output_limits():
    pid = PIDController(1.0, 0.0, 0.0, output_limit=0.5, slew_limit=0.1)
    outputs = [pid.update(100.0)[0] for _ in range(30)]
    assert abs(outputs[0]) <= 0.1
    assert all(abs(value) <= 0.5 for value in outputs)
    assert all(abs(b - a) <= 0.1 for a, b in pairwise(outputs))


def test_pair_selection_prefers_tracked_lane_over_adjacent_lane():
    components = [
        {"x": 40.0, "area": 100.0},
        {"x": 170.0, "area": 100.0},
        {"x": 285.0, "area": 100.0},
        {"x": 515.0, "area": 100.0},
    ]
    pair = select_lane_pair(components, 400.0, 230.0, 120.0, 560.0, 100.0)
    assert pair is not None
    assert pair[2] == 400.0


def test_single_line_fallback_and_jump_rejection():
    assert estimate_center_from_single(285.0, "left", 230.0, 400.0, 20.0) == 400.0
    assert estimate_center_from_single(100.0, "left", 230.0, 400.0, 20.0) is None
    with pytest.raises(ValueError):
        estimate_center_from_single(0.0, "middle", 230.0, 0.0, 20.0)
