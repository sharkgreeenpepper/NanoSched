"""Metrics derived from actual CPU-visible token events, in seconds."""
import numpy as np


def percentiles(values):
    return {f"p{p}": float(np.percentile(values, p)) if values else None for p in (50, 95, 99)}


def summarize(requests, elapsed, ttft_slo, tpot_slo):
    ttft, tpot, e2e, gaps = [], [], [], []
    good = tokens = tpot_met = 0
    for r in requests:
        events = r["token_times"]
        if not events:
            continue
        first = events[0] - r["submitted_at"]
        latency = r["finished_at"] - r["submitted_at"]
        intervals = np.diff(events).tolist()
        per_token = float(np.mean(intervals)) if intervals else None
        ttft.append(first)
        e2e.append(latency)
        gaps.extend(intervals)
        if per_token is not None:
            tpot.append(per_token)
        good += first <= ttft_slo and (per_token is None or per_token <= tpot_slo)
        tpot_met += per_token is None or per_token <= tpot_slo
        tokens += len(events)
    return {
        "requests": len(requests), "output_tokens": tokens, "elapsed_s": elapsed,
        "ttft_s": percentiles(ttft), "tpot_s": percentiles(tpot),
        "inter_token_gap_s": percentiles(gaps), "e2e_s": percentiles(e2e),
        "output_tokens_per_s": tokens / elapsed,
        "requests_per_s": len(requests) / elapsed,
        "slo_goodput_requests_per_s": good / elapsed,
        "slo_met_requests": int(good), "slo_ttft_s": ttft_slo, "slo_tpot_s": tpot_slo,
        "tpot_slo_met_fraction": tpot_met / len(requests) if requests else 0,
        "itl_slo_met_fraction": sum(g <= tpot_slo for g in gaps) / len(gaps) if gaps else 1,
    }
