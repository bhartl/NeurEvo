from unittest import TestCase


class TestRNNModule(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import Recurrent
        if verbose:
            print(Recurrent)
