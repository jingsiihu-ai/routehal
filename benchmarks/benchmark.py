"""Small, reproducible CPU benchmark for the NumPy RouteHal core."""

from __future__ import annotations

import platform
import statistics
import time

import numpy as np

from routehal import compute_hroute


def main() -> None:
    rng = np.random.default_rng(17)
    layers, batch, patches, experts = 16, 32, 196, 8
    visual = rng.dirichlet(np.ones(experts), size=(layers, batch, patches))
    attention = rng.dirichlet(np.ones(patches), size=(layers, batch))
    text = rng.dirichlet(np.ones(experts), size=(layers, batch))

    for _ in range(5):
        compute_hroute(text, visual, attention)

    timings_ms: list[float] = []
    for _ in range(50):
        start = time.perf_counter()
        compute_hroute(text, visual, attention)
        timings_ms.append((time.perf_counter() - start) * 1_000)

    p95 = sorted(timings_ms)[int(0.95 * len(timings_ms)) - 1]
    input_mib = (visual.nbytes + attention.nbytes + text.nbytes) / (1024**2)
    print(f"python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"shape: layers={layers}, batch={batch}, patches={patches}, experts={experts}")
    print(f"median_ms: {statistics.median(timings_ms):.3f}")
    print(f"p95_ms: {p95:.3f}")
    print(f"input_array_mib: {input_mib:.3f}")


if __name__ == "__main__":
    main()
