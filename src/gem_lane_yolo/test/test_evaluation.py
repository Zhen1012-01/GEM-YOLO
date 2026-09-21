import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[3] / "tools" / "evaluate_run.py"
SPEC = importlib.util.spec_from_file_location("evaluate_run", MODULE_PATH)
evaluate_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_run)


def test_summary_computes_metrics_without_fabricated_samples():
    rows = [
        {"lane_error_px": "10", "steering": "0.1", "lane_detected": "true"},
        {"lane_error_px": "20", "steering": "0.62", "lane_detected": "false"},
    ]
    result = evaluate_run.summarize(rows, 0.62)
    assert result["mean_absolute_lane_error_px"] == 15
    assert result["lane_loss_count"] == 1
    assert result["steering_saturation_count"] == 1
