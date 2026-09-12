# Reproducible results

These are sanity-check results for the public NumPy implementation. They are not end-to-end VLM accuracy claims.

## Deterministic synthetic check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/synthetic_demo.py
```

```text
aligned score:    0.0
mismatched score: 0.4068
```

The fixed-seed example keeps one token aligned with its attention-weighted visual routing prototype and deliberately routes the other token toward a different expert. The mismatch therefore receives the higher score.

## CPU microbenchmark

```bash
PYTHONPATH=src python benchmarks/benchmark.py
```

Reference run on Linux x86_64, Intel Xeon Platinum 8573C, Python 3.12.14, and NumPy 2.3.5:

| Workload | Median | p95 | Input arrays |
| --- | ---: | ---: | ---: |
| 16 layers × 32 items × 196 patches × 8 experts | 13.040 ms | 14.688 ms | 6.922 MiB |

The script performs five warmups and reports 50 single-call measurements. `Input arrays` is the exact NumPy `nbytes` total for the three inputs, not process peak memory. Runtime is a point measurement on shared cloud hardware and should be rerun on the target machine.

## Tests

```bash
python -m unittest discover -s tests -v
```

The current suite contains seven tests covering divergence bounds and symmetry, attention-weighted prototype construction, mismatch ranking, layer weighting, calibration, and input validation.

## Evaluation boundary

A meaningful model-level evaluation still requires a public MoE vision-language model, a documented hallucination dataset, model-specific router and attention hooks, and task-level metrics such as AUROC or calibration error. Those results are intentionally not fabricated here.
