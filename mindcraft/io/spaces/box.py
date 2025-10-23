import numpy as np
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


class Box(spaces.Box):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    * Independent bound for each dimension::
        >>> Box([-1.0, -2.0], high=[2.0, 4.0], dtype=np.float32)
        Box(2,)

    """
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if not np.isscalar(low):
            low = np.asarray(low, dtype=dtype)

        if not np.isscalar(high):
            high = np.asarray(high, dtype=dtype)

        super(Box, self).__init__(low, high, shape=shape, dtype=dtype)

    def __repr__(self):
        if len(self.low) > 1:
            return "Box({}, {}, {}, np.{})".format(self.low.tolist(), self.high.tolist(), self.shape, self.dtype)

        return "Box({}, {}, {}, np.{})".format(self.low[0], self.high[0], self.shape, self.dtype)
