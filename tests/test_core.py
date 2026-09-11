import numpy as np
import unittest

from routehal import compute_hroute, js_divergence, visual_routing_prototype


class RouteHalCoreTests(unittest.TestCase):
    def test_js_divergence_is_zero_for_identical_distributions(self):
        p = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(js_divergence(p, p), 0.0, atol=1e-7)

    def test_js_divergence_is_symmetric_and_bounded(self):
        p = np.array([[0.95, 0.05]])
        q = np.array([[0.05, 0.95]])
        forward = js_divergence(p, q)
        backward = js_divergence(q, p)
        np.testing.assert_allclose(forward, backward)
        self.assertTrue(0.0 <= float(forward[0]) <= 1.0)

    def test_visual_prototype_follows_attention(self):
        routing = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
        attention = np.array([[[0.75, 0.25]]])
        expected = np.array([[[0.75, 0.25]]])
        np.testing.assert_allclose(visual_routing_prototype(routing, attention), expected, atol=1e-7)

    def test_hroute_ranks_mismatch_above_alignment(self):
        visual = np.array([[[[0.9, 0.1], [0.8, 0.2]], [[0.9, 0.1], [0.8, 0.2]]]])
        attention = np.array([[[0.5, 0.5], [0.5, 0.5]]])
        text = np.array([[[0.85, 0.15], [0.05, 0.95]]])
        result = compute_hroute(text, visual, attention)
        self.assertLess(result.per_example[0], result.per_example[1])

    def test_invalid_shapes_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "visual_router_probs"):
            visual_routing_prototype(np.ones((2, 3, 4)), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
