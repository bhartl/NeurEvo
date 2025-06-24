from unittest import TestCase


class TestMindcraftModule(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import Patchwork
        if verbose:
            print(Patchwork)
