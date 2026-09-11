# RouteHal

RouteHal is a small, framework-agnostic research prototype for measuring whether a generated token's Mixture-of-Experts routing pattern is aligned with the visual evidence it attends to.

The repository implements the core `H_route` signal described in *RouteHal: Expert Routing Divergence as a Hallucination Signal in MoE Vision-Language Models*:

1. Build an attention-weighted routing prototype from visual tokens.
2. Compare a generation-step routing distribution with that prototype using normalized Jensen-Shannon divergence.
3. Aggregate the divergence across selected MoE layers.
4. Optionally standardize the score with calibration statistics and use it as an adaptive decoding gate.

## Why this repository exists

The code is intentionally compact and model-agnostic. It isolates the numerical method from any one VLM implementation, so the behavior can be tested with synthetic tensors before wiring it into a model-specific hook or generation loop.

This is a clean-room public implementation. It does not contain employer code, private data, internal endpoints, or proprietary model integrations. It also does not claim the placeholder experimental values from earlier research drafts as validated results.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/synthetic_demo.py
```

Expected behavior: a token whose routing matches the attended visual prototype receives a low score, while a deliberately mismatched token receives a higher score.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Core API

```python
from routehal import compute_hroute

result = compute_hroute(
    text_router_probs=text_router_probs,      # [layers, batch, experts]
    visual_router_probs=visual_router_probs,  # [layers, batch, patches, experts]
    visual_attention=visual_attention,        # [layers, batch, patches]
)

print(result.per_example)
print(result.per_layer)
```

All probability tensors are normalized defensively. The implementation supports NumPy arrays and validates shapes, finite values, and non-negative inputs.

## Model integration

A production integration typically needs to capture two model-internal signals during generation:

- full router probability vectors before sparse top-k dispatch;
- attention from the current prediction state to visual-token positions.

`routehal.hooks.RouterCache` provides a minimal adapter for frameworks whose modules expose forward hooks. Model-specific token indexing, attention extraction, and decoding intervention are deliberately left outside the numerical core.

## Scope and limitations

- `H_route` is a diagnostic signal, not a proof of causal expert specialization.
- Calibration may shift across models, datasets, and decoding settings.
- This repository does not bundle model weights or benchmark data.
- End-to-end MoE-VLM evaluation remains model-specific.

## License

MIT
