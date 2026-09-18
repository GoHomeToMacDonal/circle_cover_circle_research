import io
import json

from core_shape_experiment import run_core_experiment
from finite_cover import Budget


def test_core_shape_experiment_has_budget_progress_and_incremental_output(tmp_path):
    output = tmp_path / "core.json"
    progress = io.StringIO()
    payload = run_core_experiment(
        [0.99], [0], [0.05], [1.0],
        Budget(time_limit=5.0, max_nodes=0, max_candidates=100_000,
               progress_interval=0.001),
        output,
        progress,
    )
    case = payload["cases"][0]
    assert case["points"] == 256
    assert case["outer_points"] == case["core_points"] == 128
    assert case["known_25_owner_cover"]["covered"]
    assert case["candidate_family"]["status"] == "complete"
    assert case["candidate_family"]["maximal_masks"] > 0
    assert case["cover"]["status"] == "unknown"
    assert case["cover"]["stop_reason"] == "node_limit"
    assert case["cover"]["upper_bound"] <= 25
    assert json.loads(output.read_text())["cases"][0]["core_scale"] == 0.05
    text = progress.getvalue()
    assert '"stage": "case_start"' in text
    assert '"stage": "candidate_generation"' in text
    assert '"stage": "case_done"' in text
