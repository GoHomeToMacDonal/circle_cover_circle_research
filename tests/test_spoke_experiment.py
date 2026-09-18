import io
import json

from finite_cover import Budget
from spoke_experiment import run_spoke_case, run_spoke_experiment


def test_spoke_case_records_layers_and_respects_node_budget():
    progress = io.StringIO()
    case = run_spoke_case(
        0.99, 0, "single05", (0.05,), 1.0,
        Budget(time_limit=5.0, max_nodes=0, max_candidates=100_000,
               progress_interval=0.001),
        progress,
    )
    assert case["points"] == 256
    assert case["layer_counts"] == {"outer": 128, "0.05": 128}
    assert case["known_25_owner_cover"]["covered"]
    assert case["candidate_family"]["status"] == "complete"
    assert case["cover"]["status"] == "unknown"
    assert case["cover"]["stop_reason"] == "node_limit"
    assert case["cover"]["upper_bound"] <= 25
    assert '"stage": "candidate_generation"' in progress.getvalue()


def test_spoke_experiment_persists_partial_candidate_limit_result(tmp_path):
    output = tmp_path / "spokes.json"
    progress = io.StringIO()
    payload = run_spoke_experiment(
        [0.99], [0], ["small3"], [0.99],
        Budget(time_limit=5.0, max_nodes=100, max_candidates=100,
               progress_interval=0.001),
        output,
        progress,
    )
    case = payload["cases"][0]
    assert case["points"] == 512
    assert case["candidate_family"]["status"] == "unknown"
    assert case["candidate_family"]["stop_reason"] == "candidate_limit"
    assert case["cover"]["status"] == "unknown"
    assert json.loads(output.read_text())["cases"][0]["schedule"] == "small3"
    text = progress.getvalue()
    assert '"stage": "case_start"' in text
    assert '"stage": "case_done"' in text
