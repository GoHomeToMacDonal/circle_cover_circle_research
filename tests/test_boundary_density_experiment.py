import io
import json

from boundary_density_experiment import run_experiment
from finite_cover import Budget


def test_budgeted_density_experiment_records_progress_and_partial_output(tmp_path):
    output = tmp_path / "density.json"
    progress = io.StringIO()
    payload = run_experiment(
        [0.99], [0], [1.0],
        Budget(time_limit=5.0, max_nodes=0, max_candidates=20_000,
               progress_interval=0.001),
        output,
        progress,
    )
    case = payload["cases"][0]
    assert case["points"] == 128
    assert case["known_25_owner_cover"]["covered"]
    assert case["candidate_family"]["status"] == "complete"
    assert case["cover"]["status"] == "unknown"
    assert case["cover"]["stop_reason"] == "node_limit"
    assert case["cover"]["upper_bound"] <= 25
    assert output.exists()
    persisted = json.loads(output.read_text())
    assert persisted["cases"][0]["points"] == 128
    text = progress.getvalue()
    assert '"stage": "case_start"' in text
    assert '"stage": "candidate_generation"' in text
    assert '"stage": "case_done"' in text
