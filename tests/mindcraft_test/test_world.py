from unittest import TestCase


class TestWorld(TestCase):
    def test_import(self, verbose=False):
        from mindcraft import World
        if verbose:
            print(World)
