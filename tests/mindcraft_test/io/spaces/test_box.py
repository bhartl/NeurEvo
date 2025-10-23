from unittest import TestCase, skip
from numpy import *


class BoxTest(TestCase):
    def test_gym_repr_1d(self, verbose=False):
        try:
            from gymnasium.spaces import Box as GymBox
        except ImportError:
            from gym.spaces import Box as GymBox
        from mindcraft.io.spaces import Box as MCBox
        import numpy as np

        b = GymBox(-1, 1, shape=(1,))
        b_wm = MCBox(-1, 1, shape=(1,))
        self.assertEqual(b.low, -1)
        self.assertEqual(b.high, 1)

        samples = asarray([b.sample() for __ in range(10)])
        self.assertTrue(samples.min() >= -1)
        self.assertTrue(samples.max() <= 1)

        for b, cls in zip([b_wm], [MCBox],):

            # make repr
            b_repr = repr(b)
            if verbose:
                print(b_repr)

            # load from repr
            Box = cls  # necessary for eval
            b1 = eval(b_repr)
            self.assertEqual(b, b1)

    def test_gym_repr_2d(self, verbose=False):
        import numpy as np
        try:
            from gymnasium.spaces import Box as GymBox
        except ImportError:
            from gym.spaces import Box as GymBox
        from mindcraft.io.spaces import Box as MCBox
        # same low/high grid
        b = GymBox(-1, 1, shape=(2,))
        with self.assertRaises(ValueError):
            GymBox([-1., 0.], [1., 1.5])  # not allowed with lists

        b_asym = GymBox(array([-1., 0.], dtype=np.float32), array([1., 1.5], dtype=np.float32))

        # WMBox works with scalars, lists and arrays
        b_wm = MCBox(-1, 1, shape=(2,))
        b_wm_asym_from_arrays = MCBox([-1., 0.], [1., 1.5])
        b_wm_asym_from_lists = MCBox([-1., 0.], [1., 1.5])

        self.assertEqual(b, b_wm)
        self.assertEqual(b_asym, b_wm_asym_from_arrays)
        self.assertEqual(b_asym, b_wm_asym_from_lists)
        b_wm_asym = b_wm_asym_from_lists

        self.assertTrue(b.shape == (2,))

        self.assertEqual(b.low[0], -1)
        self.assertEqual(b.low[1], -1)

        self.assertEqual(b.high[0], 1)
        self.assertEqual(b.high[1], 1)

        boxes = [b, b_asym, b_wm, b_wm_asym]
        for bi, r_low, r_high in zip(boxes,
                                     [bi.low for bi in boxes],
                                     [bi.high for bi in boxes],):

            samples = asarray([bi.sample() for __ in range(10)])
            self.assertTrue(samples[:, 0].min() >= r_low[0])
            self.assertTrue(samples[:, 1].min() >= r_low[1])
            self.assertTrue(samples[:, 0].max() <= r_high[0])
            self.assertTrue(samples[:, 1].max() <= r_high[1])

        # valid representations
        for b, cls in zip([b_wm, b_wm_asym], [MCBox, MCBox],):
            # make repr
            b_repr = repr(b)
            if verbose:
                print('valid repr:', b_repr)

            # load from repr
            Box = cls
            b1 = eval(b_repr)
            self.assertEqual(b, b1)
            self.assertTrue(array_equal(b.low, b1.low))
            self.assertTrue(array_equal(b.high, b1.high))
            self.assertTrue(array_equal(b.shape, b1.shape))
