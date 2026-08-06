from __future__ import annotations

import pytest

from moa_gateway.benchmark import (
    CASES,
    BenchmarkResult,
    _execute_case,
    _extract_code,
    _percentile,
    _screen_code,
    _strategy_gate,
    _summarize_results,
)
from moa_gateway.config import load_config


def test_extracts_largest_python_fence() -> None:
    content = "before\n```python\nx = 1\n```\n```python\nx = 1\ny = 2\n```"

    assert _extract_code(content) == "x = 1\ny = 2"


def test_screen_rejects_unsafe_imports_and_calls() -> None:
    with pytest.raises(ValueError, match="disallowed import"):
        _screen_code("import subprocess")
    with pytest.raises(ValueError, match="disallowed call"):
        _screen_code("eval('1 + 1')")


def test_builtin_merge_intervals_reference_passes() -> None:
    code = """def merge_intervals(intervals):
    values = sorted(intervals)
    if any(start > end for start, end in values):
        raise ValueError("invalid interval")
    merged = []
    for start, end in values:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged
"""

    passed, error = _execute_case(code, CASES[0])

    assert passed is True
    assert error is None


def test_percentile_interpolates_p50_and_p95() -> None:
    assert _percentile([1, 2, 3, 4], 0.5) == 2.5
    assert _percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert _percentile([7], 0.95) == 7


def test_summary_reports_pass_at_1_latency_and_split_tokens() -> None:
    results = [
        BenchmarkResult("direct", "direct", "one", 1, True, 1, 10, 2),
        BenchmarkResult("direct", "direct", "two", 1, False, 3, 20, 4),
    ]

    summary = _summarize_results(results, ["direct"])["direct"]

    assert summary == {
        "passed": 1,
        "total": 2,
        "pass_at_1": 0.5,
        "p50_latency_seconds": 2.0,
        "p95_latency_seconds": 2.9,
        "total_input_tokens": 30,
        "total_output_tokens": 6,
    }


def test_strategy_gate_requires_strict_quality_improvement() -> None:
    summaries = {
        "direct": {
            "pass_at_1": 0.8,
            "p95_latency_seconds": 2,
            "total_input_tokens": 100,
            "total_output_tokens": 20,
        },
        "tie": {
            "pass_at_1": 0.8,
            "p95_latency_seconds": 1,
            "total_input_tokens": 50,
            "total_output_tokens": 10,
        },
        "better": {
            "pass_at_1": 0.9,
            "p95_latency_seconds": 4,
            "total_input_tokens": 200,
            "total_output_tokens": 30,
        },
    }

    gate = _strategy_gate(summaries)

    assert gate["selected"] == "better"
    summaries["better"]["pass_at_1"] = 0.8
    assert _strategy_gate(summaries)["selected"] == "direct"


def test_benchmark_profiles_cover_four_strategy_arms() -> None:
    config = load_config("moa.yaml")

    k2 = config.flows["council-k2"].step_map
    k3 = config.flows["council-k3"].step_map
    assert k2["proposals"].min_success == 2
    assert k3["proposals"].min_success == 3
    assert {step_id for step_id in k2 if step_id in {"implementer", "reviewer"}} == {
        "implementer",
        "reviewer",
    }
    assert {k3[name].model for name in ("implementer", "reviewer", "edge-cases")} == {
        "qwen2.5-coder:7b",
        "gemma4:latest",
        "deepseek-coder-v2:16b",
    }
    self_models = {
        config.flows["self-consistency"].step_map[name].model
        for name in ("sample-one", "sample-two", "sample-three")
    }
    assert self_models == {"qwen2.5-coder:7b"}
