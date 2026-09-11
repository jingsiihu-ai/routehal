"""Demonstrate that routing mismatch produces a larger RouteHal score."""

import numpy as np

from routehal import compute_hroute


rng = np.random.default_rng(7)
layers, batch, patches, experts = 4, 2, 8, 4

visual_routing = rng.dirichlet(np.ones(experts), size=(layers, batch, patches))
attention = rng.dirichlet(np.ones(patches), size=(layers, batch))

prototype = np.sum(attention[..., None] * visual_routing, axis=2)
text_routing = prototype.copy()

# Keep the first example aligned and make the second route to a different expert.
text_routing[:, 1, :] = np.array([0.01, 0.01, 0.97, 0.01])

result = compute_hroute(text_routing, visual_routing, attention)
print("aligned score:   ", round(float(result.per_example[0]), 4))
print("mismatched score:", round(float(result.per_example[1]), 4))
