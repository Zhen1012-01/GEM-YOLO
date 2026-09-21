"""ROS-independent lane-following control primitives."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to an inclusive range."""
    if lower > upper:
        raise ValueError("lower limit must not exceed upper limit")
    return max(lower, min(value, upper))


@dataclass
class PIDController:
    """PID controller with integral, output, and slew-rate limits."""

    kp: float
    ki: float
    kd: float
    output_limit: float
    slew_limit: float
    integral_limit: float = 1500.0
    derivative_alpha: float = 0.35
    integral: float = 0.0
    previous_error: float = 0.0
    previous_derivative: float = 0.0
    previous_output: float = 0.0

    def update(self, error: float) -> tuple[float, float]:
        self.integral = clamp(
            self.integral + error, -self.integral_limit, self.integral_limit
        )
        raw_derivative = error - self.previous_error
        derivative = (
            self.derivative_alpha * raw_derivative
            + (1.0 - self.derivative_alpha) * self.previous_derivative
        )
        target = clamp(
            -(self.kp * error + self.ki * self.integral + self.kd * derivative),
            -self.output_limit,
            self.output_limit,
        )
        delta = clamp(target - self.previous_output, -self.slew_limit, self.slew_limit)
        output = clamp(
            0.68 * self.previous_output + 0.32 * (self.previous_output + delta),
            -self.output_limit,
            self.output_limit,
        )
        self.previous_error = error
        self.previous_derivative = derivative
        self.previous_output = output
        return output, derivative

    def reset_soft(self) -> None:
        self.integral = 0.0
        self.previous_derivative = 0.0


def select_lane_pair(
    components: Iterable[Mapping[str, float]],
    predicted_center: float,
    lane_width: float,
    min_width: float,
    max_width: float,
    max_center_jump: float,
) -> tuple[Mapping[str, float], Mapping[str, float], float, float] | None:
    """Select the boundary pair nearest the tracked lane, rejecting adjacent lanes."""
    items = list(components)
    expected_left = predicted_center - lane_width / 2.0
    expected_right = predicted_center + lane_width / 2.0
    candidates = []
    for left in items:
        for right in items:
            if left is right or left["x"] >= right["x"]:
                continue
            width = right["x"] - left["x"]
            if not min_width <= width <= max_width:
                continue
            center = (left["x"] + right["x"]) / 2.0
            center_distance = abs(center - predicted_center)
            if center_distance > max_center_jump:
                continue
            edge_distance = abs(left["x"] - expected_left) + abs(
                right["x"] - expected_right
            )
            area_bonus = min((left.get("area", 0) + right.get("area", 0)) / 500, 8)
            candidates.append(
                (
                    center_distance + 0.35 * edge_distance - area_bonus,
                    left,
                    right,
                    center,
                    width,
                )
            )
    if not candidates:
        return None
    _, left, right, center, width = min(candidates, key=lambda item: item[0])
    return left, right, center, width


def estimate_center_from_single(
    line_x: float,
    side: str,
    lane_width: float,
    predicted_center: float,
    max_center_jump: float,
) -> float | None:
    """Infer lane center from one observed boundary, with a jump guard."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    center = line_x + lane_width / 2.0 if side == "left" else line_x - lane_width / 2.0
    return center if abs(center - predicted_center) <= max_center_jump else None
