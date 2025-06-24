from unittest import TestCase


class TestBONES(TestCase):
    def test_mating(self):
        import numpy as np
        from numpy import copy

        def get_random_indices(p):
            return np.random.rand(*tuple(p.shape)) > 0.5

        def mate(p_1, p_2, p_1_params):
            child_params = copy(p_2)
            for o, op_1_params in zip(range(len(child_params)), p_1_params):
                child_params[o, op_1_params] = p_1[o, op_1_params]
            return child_params

        def mate_fast(p_1, p_2, p_1_hook):
            child_params = copy(p_2)
            child_params[p_1_hook] = p_1[p_1_hook]
            return child_params

        def mate_fast2(p_1, p_2, p_1_hook):
            child_params = copy(p_2)
            p_1_hook = np.where(p_1_hook)
            child_params[p_1_hook] = p_1[p_1_hook]
            return child_params

        p1, p2 = np.random.randn(2, 100, 20)

        r = get_random_indices(p1)
        self.assertTrue(np.any(r))
        self.assertTrue(not np.all(r))

        m1 = mate(p1, p2, r)
        m2 = mate_fast(p1, p2, r)
        m3 = mate_fast2(p1, p2, r)

        # check whether correctly copied p1[r] -> m1[r] and p2[~r] -> m1[~r]
        self.assertTrue(np.array_equal(m1[r], p1[r]))
        self.assertTrue(np.array_equal(m1[~r], p2[~r]))

        # check whether same results
        self.assertTrue(np.array_equal(m1, m2))
        self.assertTrue(np.array_equal(m1, m3))

        # make different choice, should give different crossover
        r1 = get_random_indices(p1)
        m4 = mate_fast(p1, p2, r1)
        self.assertFalse(np.array_equal(m1, m4))
