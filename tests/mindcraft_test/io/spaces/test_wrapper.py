from unittest import TestCase


class WrapperTest(TestCase):
    def test_clip_discrete(self):
        from mindcraft.io.spaces import space_clip
        from mindcraft.io.spaces import Discrete
        import numpy as np

        n = 5
        d = Discrete(n)
        randoms = np.random.randint(-4, 10, size=100)
        clipped = space_clip(randoms, d)

        randoms_individual = np.asarray([(r - 0.5) * 100. for r in np.random.rand(100)])
        clipped_individual = np.asarray([space_clip(r, d) for r in randoms_individual])

        self.assertTrue((0 <= clipped).all())
        self.assertTrue((clipped < n).all())
        valid_indices = (0 <= randoms) & (n > randoms)
        self.assertTrue(np.array_equal(randoms[valid_indices], clipped[valid_indices]))

        self.assertTrue((0 <= clipped_individual).all())
        self.assertTrue((clipped_individual < n).all())
        vi = (0 <= np.round(randoms_individual)) & (n > np.round(randoms_individual))
        self.assertFalse(np.array_equal(randoms_individual[vi], clipped_individual[vi])), "Cast To Int"
        self.assertTrue(np.array_equal(np.round(randoms_individual[vi]), clipped_individual[vi]))

    def test_clip_box(self):
        from mindcraft.io.spaces import space_clip
        from mindcraft.io.spaces import Box
        import numpy as np

        low = [-10., 0.5, 0.]
        high = [10., 11.5, 110.]
        b = Box(low=low, high=high)

        randoms = (np.random.randn(100, 3) - 0.5) * 200.
        clipped = space_clip(randoms, b)

        for c, r in zip(clipped, randoms):
            self.assertTrue((c >= low).all())
            self.assertTrue((c <= high).all())

            valid_indices = (low <= r) & (high >= r)
            self.assertTrue(np.array_equal(r[valid_indices], c[valid_indices]))
