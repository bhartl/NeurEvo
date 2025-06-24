from unittest import TestCase


class TestMindcraft(TestCase):
    def test_import(self, verbose=False):
        import mindcraft
        if verbose:
            print(mindcraft)
