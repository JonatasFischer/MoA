from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from moa_gateway.config import GatewayConfig
from moa_gateway.domain import CanonicalRequest
from moa_gateway.gateway import Gateway


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    prompt: str
    tests: str


@dataclass(slots=True)
class BenchmarkResult:
    profile: str
    strategy: str
    case: str
    run: int
    passed: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


CASES = (
    BenchmarkCase(
        name="merge_intervals",
        prompt="""Return only one fenced Python code block. Implement
merge_intervals(intervals), where intervals is an iterable of (start, end)
integer pairs. Return a new sorted list of merged tuples. Overlapping or
touching closed intervals must merge. Reject start > end with ValueError and do
not mutate the input.""",
        tests="""from solution import merge_intervals

assert merge_intervals([]) == []
assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]
assert merge_intervals([(5, 7), (1, 3), (3, 5)]) == [(1, 7)]
assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]
source = [(8, 9), (1, 2)]
assert merge_intervals(source) == [(1, 2), (8, 9)]
assert source == [(8, 9), (1, 2)]
try:
    merge_intervals([(3, 1)])
except ValueError:
    pass
else:
    raise AssertionError("invalid intervals must raise ValueError")
""",
    ),
    BenchmarkCase(
        name="topological_order",
        prompt="""Return only one fenced Python code block. Implement
topological_order(graph), where graph maps each node to an iterable of its
prerequisites. Include nodes that occur only as prerequisites. Return a
deterministic order in which every prerequisite precedes its dependent, using
lexicographic order whenever several nodes are available. Raise ValueError for
a cycle and do not mutate graph.""",
        tests="""from solution import topological_order

graph = {"build": ["compile", "lint"], "compile": ["parse"], "lint": ["parse"]}
assert topological_order(graph) == ["parse", "compile", "lint", "build"]
assert graph == {"build": ["compile", "lint"], "compile": ["parse"], "lint": ["parse"]}
assert topological_order({"b": [], "a": []}) == ["a", "b"]
assert topological_order({"deploy": ["package"]}) == ["package", "deploy"]
try:
    topological_order({"a": ["b"], "b": ["a"]})
except ValueError:
    pass
else:
    raise AssertionError("cycles must raise ValueError")
""",
    ),
    BenchmarkCase(
        name="lru_cache",
        prompt="""Return only one fenced Python code block. Implement an
LRUCache class with constructor LRUCache(capacity), get(key) returning -1 for a
miss, and put(key, value). Both operations must be O(1). Updating a key makes it
most recently used. Capacity must be positive or the constructor raises
ValueError.""",
        tests="""from solution import LRUCache

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(1, 10)
cache.put(4, 4)
assert cache.get(3) == -1
assert cache.get(1) == 10
assert cache.get(4) == 4
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("non-positive capacity must raise ValueError")
""",
    ),
)

ALLOWED_IMPORTS = {"bisect", "collections", "dataclasses", "heapq", "typing"}
BLOCKED_CALLS = {"__import__", "compile", "eval", "exec", "open"}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(content: str) -> str:
    blocks = FENCE.findall(content)
    return max(blocks, key=len).strip() if blocks else content.strip()


def _screen_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(module not in ALLOWED_IMPORTS for module in modules):
                raise ValueError(f"disallowed import: {', '.join(modules)}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in ALLOWED_IMPORTS:
                raise ValueError(f"disallowed import: {module}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BLOCKED_CALLS
        ):
            raise ValueError(f"disallowed call: {node.func.id}")


def _execute_case(code: str, case: BenchmarkCase) -> tuple[bool, str | None]:
    try:
        _screen_code(code)
    except (SyntaxError, ValueError) as exc:
        return False, f"code screening failed: {exc}"

    with tempfile.TemporaryDirectory(prefix="moa-benchmark-") as directory:
        root = Path(directory)
        (root / "solution.py").write_text(code, encoding="utf-8")
        test_path = root / "test_solution.py"
        test_path.write_text(case.tests, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "tests timed out"
        if completed.returncode == 0:
            return True, None
        detail = (completed.stderr or completed.stdout).strip()
        return False, detail[-1000:] or f"tests exited {completed.returncode}"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summarize_results(
    results: list[BenchmarkResult], profiles: list[str]
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for profile_name in profiles:
        selected = [result for result in results if result.profile == profile_name]
        passed = sum(result.passed for result in selected)
        summaries[profile_name] = {
            "passed": passed,
            "total": len(selected),
            "pass_at_1": round(passed / len(selected), 3),
            "p50_latency_seconds": round(
                _percentile([result.latency_seconds for result in selected], 0.5), 3
            ),
            "p95_latency_seconds": round(
                _percentile([result.latency_seconds for result in selected], 0.95), 3
            ),
            "total_input_tokens": sum(result.input_tokens for result in selected),
            "total_output_tokens": sum(result.output_tokens for result in selected),
        }
    return summaries


def _strategy_gate(
    summaries: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    if "direct" not in summaries:
        raise ValueError("strategy gate requires a direct baseline")
    baseline = summaries["direct"]
    baseline_quality = float(baseline["pass_at_1"])
    eligible = [
        (name, summary)
        for name, summary in summaries.items()
        if name != "direct" and float(summary["pass_at_1"]) > baseline_quality
    ]
    selected = "direct"
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                -float(item[1]["pass_at_1"]),
                float(item[1]["p95_latency_seconds"]),
                int(item[1]["total_input_tokens"])
                + int(item[1]["total_output_tokens"]),
            ),
        )[0]
    return {
        "policy": (
            "promote only a strict pass@1 improvement; then minimize p95 latency "
            "and total tokens"
        ),
        "baseline": "direct",
        "selected": selected,
        "deltas": {
            name: {
                "pass_at_1": round(float(summary["pass_at_1"]) - baseline_quality, 3),
                "p95_latency_seconds": round(
                    float(summary["p95_latency_seconds"])
                    - float(baseline["p95_latency_seconds"]),
                    3,
                ),
                "total_tokens": (
                    int(summary["total_input_tokens"])
                    + int(summary["total_output_tokens"])
                    - int(baseline["total_input_tokens"])
                    - int(baseline["total_output_tokens"])
                ),
            }
            for name, summary in summaries.items()
            if name != "direct"
        },
    }


async def run_benchmark(
    config: GatewayConfig,
    profiles: list[str],
    runs: int,
    output: Path,
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    for profile in profiles:
        if config.uses_flows:
            config.resolve_flow(profile)
        else:
            config.resolve_profile(profile)

    gateway = Gateway(config)
    results: list[BenchmarkResult] = []
    try:
        for run in range(1, runs + 1):
            for case in CASES:
                for profile_name in profiles:
                    if config.uses_flows:
                        config.resolve_flow(profile_name)
                        strategy = profile_name
                    else:
                        _, profile = config.resolve_profile(profile_name)
                        strategy = profile.strategy
                    request = CanonicalRequest(
                        requested_model=profile_name,
                        messages=[{"role": "user", "content": case.prompt}],
                        max_tokens=1024,
                        temperature=0.1,
                    )
                    started = time.perf_counter()
                    try:
                        completion = await gateway.complete(request)
                        passed, error = _execute_case(
                            _extract_code(completion.content), case
                        )
                        usage = completion.panel_usage or completion.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                    except Exception as exc:
                        passed = False
                        error = f"{type(exc).__name__}: {exc}"
                        input_tokens = 0
                        output_tokens = 0
                    result = BenchmarkResult(
                        profile=profile_name,
                        strategy=strategy,
                        case=case.name,
                        run=run,
                        passed=passed,
                        latency_seconds=round(time.perf_counter() - started, 3),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error=error,
                    )
                    results.append(result)
                    status = "PASS" if passed else "FAIL"
                    print(
                        f"{status} {profile_name}/{case.name} "
                        f"{result.latency_seconds:.3f}s "
                        f"{input_tokens + output_tokens} tokens",
                        flush=True,
                    )
    finally:
        await gateway.close()

    summaries = _summarize_results(results, profiles)
    payload: dict[str, object] = {
        "profiles": profiles,
        "runs": runs,
        "cases": [case.name for case in CASES],
        "summary": summaries,
        "strategy_gate": _strategy_gate(summaries),
        "results": [asdict(result) for result in results],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
