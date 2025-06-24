from unittest import TestCase
from itertools import product as it


class TestFeedForward(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import FeedForward
        if verbose:
            print(FeedForward)

    def test_serialize_deserialize(self):
        from mindcraft.torch.module import FeedForward
        from torch import Tensor
        from numpy import ndarray, array_equal
        from numpy.random import choice
        from mindcraft.torch.util import tensor_to_numpy

        input_size = range(1, 21, 2)
        output_size = range(1, 201, 50)
        hidden_size = range(1, 21, 5)

        for i, o in it(input_size, output_size):
            for h in range(len(hidden_size) + 1):
                hs = None
                if h == 1:
                    hs = choice(hidden_size)
                elif h > 1:
                    hs = list(hidden_size)[0:h]

                f = FeedForward(input_size=i, output_size=o, hidden_size=hs)

                s = f.serialize_parameters()
                self.assertIs(type(s), ndarray)

                r = f.recover_indices
                m = f.serialize_mask

                f1 = FeedForward(input_size=i, output_size=o, hidden_size=hs,
                                 serialized=s, recover_indices=r, serialize_mask=m)
                s1 = f1.serialize_parameters(to_numpy=False)
                self.assertIs(type(s1), Tensor)
                self.assertTrue(array_equal(s, tensor_to_numpy(s1)))

                f2 = FeedForward.from_dict(f.to_dict())
                s2 = f2.serialize_parameters(to_numpy=True)
                self.assertIs(type(s2), ndarray)
                self.assertTrue(array_equal(s, s2))
