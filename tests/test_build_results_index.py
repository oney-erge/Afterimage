import json

from scripts.build_results_index import row_for


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_row_for_extracts_core_fields(tmp_path):
    path = _write(tmp_path, "x.json", {
        "hypothesis_id": "h9-ram-overlay-head", "status": "complete",
        "evidence_level": "L1_mechanism_screen", "model": "Qwen/Qwen3-14B"})
    line = row_for(path)
    assert "h9-ram-overlay-head" in line
    assert "complete" in line
    assert "L1 mechanism screen" in line
    assert "Qwen3-14B" in line


def test_row_for_flags_interrupted_and_failed_runs(tmp_path):
    interrupted = _write(tmp_path, "a.json", {"status": "running"})
    assert "(interrupted)" in row_for(interrupted)

    failed = _write(tmp_path, "b-failed.json", {"status": "complete"})
    assert "(failed run)" in row_for(failed)


def test_row_for_defaults_missing_fields_to_dash(tmp_path):
    path = _write(tmp_path, "minimal.json", {})
    line = row_for(path)
    assert line.count(" - ") >= 3


def test_row_for_returns_none_for_unparseable_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert row_for(path) is None
