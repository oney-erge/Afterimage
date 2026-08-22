from scripts.run_regulated_pair import PROTOCOLS, _analyse


def _row(block, case, seconds, token_ids, **extra):
    row = {
        "block": block, "case_id": case, "wall_seconds": seconds,
        "output_tokens": len(token_ids), "output_token_ids": token_ids,
        "peak_vram_gb": 4.0, "bytes_read": 100,
        "storage_read_calls": 100, "storage_extent_bytes": 100,
        "prefetch_wait_seconds": 1.0,
        "prefetch_peak_inflight_bytes": 10,
    }
    row.update(extra)
    return row


def test_h12_analysis_requires_mechanism_gate_not_just_speed():
    state = {
        "read_posterior": {"count": 200},
        "lead_window_posterior": {"count": 200},
        "brier_score": 0.1,
    }
    result = {"trials": [
        {"arm": "control", "rows": [_row(0, "a", 10.0, [1])]},
        {"arm": "candidate", "rows": [
            _row(0, "a", 8.0, [1], prefetch_wait_seconds=1.1,
                 prefetch_controller_state=state)]},
    ]}
    analysis = _analyse(result, PROTOCOLS["H12"])
    assert analysis["paired_effect"]["median_speedup_effect"] > 0.05
    assert not analysis["mechanism_gate"]["passed"]
    assert not analysis["advance_to_l3"]


def test_h14_analysis_checks_calls_bytes_and_tokens():
    result = {"trials": [
        {"arm": "control", "rows": [_row(0, "a", 10.0, [1])]},
        {"arm": "candidate", "rows": [
            _row(0, "a", 8.0, [1], storage_read_calls=40,
                 storage_extent_bytes=103)]},
    ]}
    analysis = _analyse(result, PROTOCOLS["H14"])
    assert analysis["mechanism_gate"]["passed"]
    assert analysis["paired_token_exact"]
