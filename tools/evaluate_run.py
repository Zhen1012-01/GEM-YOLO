#!/usr/bin/env python3
"""Summarize lane-following telemetry exported as CSV.

Expected columns: timestamp, lane_error_px, steering, lane_detected,
inference_ms, processing_ms. Missing optional latency columns are ignored.
"""

import argparse
import csv
import statistics
from itertools import pairwise
from pathlib import Path


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"lane_error_px", "steering", "lane_detected"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return rows


def summarize(rows, steering_limit):
    errors = [abs(float(row["lane_error_px"])) for row in rows if row["lane_error_px"]]
    steering = [float(row["steering"]) for row in rows]
    changes = [abs(b - a) for a, b in pairwise(steering)]
    report = {
        "frames": len(rows),
        "mean_absolute_lane_error_px": statistics.fmean(errors) if errors else None,
        "lane_loss_count": sum(
            row["lane_detected"].lower() in {"0", "false", "no"} for row in rows
        ),
        "mean_absolute_steering_change": statistics.fmean(changes) if changes else 0.0,
        "steering_saturation_count": sum(
            abs(value) >= steering_limit for value in steering
        ),
    }
    for column in ("inference_ms", "processing_ms"):
        values = [float(row[column]) for row in rows if row.get(column)]
        if values:
            report[f"mean_{column}"] = statistics.fmean(values)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--steering-limit", type=float, default=0.62)
    args = parser.parse_args()
    for key, value in summarize(load_rows(args.csv_file), args.steering_limit).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
