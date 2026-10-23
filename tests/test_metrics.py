import pytest
from benchmarks.metrics import summarize


def test_metrics_use_real_token_events_and_planned_arrival():
    rows = [{"submitted_at": 1., "enqueued_at": 1.5, "token_times": [2., 2.2, 2.6], "finished_at": 2.6},
            {"submitted_at": 0., "enqueued_at": 0., "token_times": [1.], "finished_at": 1.}]
    result = summarize(rows, 4., 1.1, 0.4)
    assert result["ttft_s"]["p50"] == 1.
    assert result["tpot_s"]["p50"] == pytest.approx(0.3)
    assert result["requests_per_s"] == 0.5
    assert result["output_tokens_per_s"] == 1.
    assert result["slo_goodput_requests_per_s"] == 0.5
