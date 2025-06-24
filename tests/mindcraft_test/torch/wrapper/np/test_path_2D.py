from unittest import TestCase


class TestPatch2D(TestCase):
    def test_import_patch2d(self, verbose=False):
        from mindcraft.torch.wrapper.np import Patch2D
        if verbose:
            print(Patch2D)
