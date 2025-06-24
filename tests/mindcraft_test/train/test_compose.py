from unittest import TestCase


class TestCompose(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.train import Compose
        if verbose:
            print(Compose)
