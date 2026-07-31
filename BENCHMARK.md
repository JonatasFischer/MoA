# Local Coding Benchmark

## Environment

- Date: 2026-07-29
- Hardware: Apple M5 Pro, 48 GB unified memory
- Runtime: Ollama 0.32.1
- Model: `qwen3-coder:30b-128k`, Q4_K_M, 30.5B total / 3.3B active
- Direct profile: one Qwen3-Coder completion
- MoA profile: two concurrent role-conditioned Qwen3-Coder proposals followed
  by one Qwen3-Coder synthesis

Using one model family for the panel shares weights and was more reliable than
mixing weaker small coding models on this machine. In a model-selection run,
Qwen3-Coder passed 3/3 tasks, DeepSeek-Coder-V2-Lite passed 1/3, and
Qwen2.5-Coder 7B passed 0/3. The mixed-model MoA passed 2/3, so it was rejected.

## Method

The built-in benchmark asks each profile to generate Python implementations for
three tasks: interval merging, deterministic topological sorting, and an O(1)
LRU cache. Generated code is parsed and screened for unsafe imports and dynamic
execution before deterministic tests run in a temporary directory.

The current harness alternates four arms over identical tasks and runs:
`direct`, `council-k2`, `council-k3`, and `self-consistency`. The repeated-sample
self-consistency arm uses three independent generations from one model followed
by synthesis; it is a benchmark arm, not a separate production strategy.

Each arm is scored only on its final answer. Reports include pass@1, p50 and p95
latency, and total input/output tokens across every model stage. The strategy
gate keeps `direct` unless another arm has strictly higher pass@1; quality
improvements are then ranked by p95 latency and total tokens. This prevents a
quality tie from promoting a much slower, more expensive strategy.

## Historical Results

| Profile | Passed | Pass rate | Mean latency | Total panel tokens |
|---|---:|---:|---:|---:|
| Direct | 9/9 | 100% | 2.601 s | 2,661 |
| Self-MoA | 9/9 | 100% | 13.017 s | 22,523 |

On this small suite, MoA matched direct quality but required approximately 5.0x
the latency and 8.5x the tokens. This proves the orchestration path works, but it
does not establish a quality advantage. Use `direct-code` for latency-sensitive
work and `moa-code` when testing aggregation on harder tasks where independent
review may justify the overhead.

Raw per-case measurements, summaries, and the promotion decision are written to
`benchmark-results.json` by the benchmark command. The historical tables below
predate the four-arm gate and are retained only as prior measurements.

## Two-Parallel Result

The benchmark was repeated after configuring Ollama with two parallel 128K
contexts, Flash Attention, and Q8 KV cache. The runner confirmed `-np 2` and
`-c 262144`; the model occupied 32 GB and remained fully GPU-resident.

| Profile | Passed | Pass rate | Mean latency | Total panel tokens |
|---|---:|---:|---:|---:|
| Direct | 9/9 | 100% | 3.650 s | 2,676 |
| Self-MoA | 9/9 | 100% | 14.754 s | 21,942 |

This run did not improve end-to-end latency over the earlier one-parallel run.
The final repetition slowed for both profiles while free memory fell to 14%, so
the mean includes system-pressure or thermal effects. Parallel inference can
increase aggregate throughput, but this MoA request still waits for both
proposers and they contend for the same GPU. Raw measurements are in
`benchmark-results-parallel-2.json`.

## Limitations

- Three synthetic tasks are too small to estimate repository-level agent quality.
- The benchmark measures code generation, not tool-call or patch-loop behavior.
- Results are local, quantization-specific, and sensitive to model residency.
- Generated code screening reduces risk but is not an operating-system sandbox.
