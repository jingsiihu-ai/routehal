"""A minimal cache for router outputs captured by model-specific forward hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RouterCache:
    """Store detached router outputs by layer without depending on PyTorch.

    The returned hook follows the common ``hook(module, inputs, output)`` shape.
    If the output implements ``detach`` and ``cpu``, both are called before the
    value is stored. Converting logits to probabilities remains the adapter's
    responsibility because router modules differ across MoE implementations.
    """

    def __init__(self) -> None:
        self.values: dict[int, Any] = {}

    def clear(self) -> None:
        self.values.clear()

    def make_hook(self, layer_index: int) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output.detach() if hasattr(output, "detach") else output
            value = value.cpu() if hasattr(value, "cpu") else value
            self.values[layer_index] = value

        return hook
