# Method notes

## Inputs

For selected MoE layers, RouteHal consumes:

- prediction-state router distributions `text_router_probs` with shape `[L, B, E]`;
- visual-token router distributions `visual_router_probs` with shape `[L, B, V, E]`;
- attention mass from the prediction state to visual tokens with shape `[L, B, V]`.

Here `L` is the number of selected layers, `B` the batch size, `V` the number of visual tokens, and `E` the number of experts.

## Attention-weighted visual prototype

For generation step `t` and layer `l`, the visual routing prototype is

```text
r_visual(l, t) = sum_j attention(l, t, j) * router(l, visual_j)
```

Attention is renormalized over the visual positions. This makes the reference prototype token-specific: the routing for a generated object is compared primarily with the image regions used to predict it.

## Routing divergence

The per-layer mismatch is normalized Jensen-Shannon divergence:

```text
D(l, t) = JS(router(l, prediction_t) || r_visual(l, t)) / log(2)
```

The division by `log(2)` maps the result to `[0, 1]` when natural logarithms are used. `compute_hroute` aggregates `D(l, t)` with uniform or user-supplied layer weights.

## Calibration

Raw divergence can reflect a general modality gap rather than hallucination alone. A held-out set of correctly grounded prediction steps may be used to estimate a mean and standard deviation. RouteHal then reports a calibrated z-score:

```text
H_calibrated(t) = (H_route(t) - mean_grounded) / std_grounded
```

Calibration statistics are model- and domain-dependent and should not be transferred blindly.

## Integration boundary

The public package intentionally stops at the model-agnostic numerical core. A model adapter is responsible for:

1. identifying visual-token positions;
2. capturing full router probabilities before sparse dispatch;
3. extracting prediction-to-visual attention;
4. selecting layers and estimating calibration statistics;
5. deciding how, or whether, a high score affects decoding.

This separation keeps the core testable without model weights and avoids claiming that routing divergence establishes causal expert specialization.

## Related ideas

The implementation is an original, compact reference informed by public work on MoE vision-language models, Jensen-Shannon divergence, visual contrastive decoding, and attention-guided hallucination mitigation. It does not copy model-specific training or evaluation code from those projects.
