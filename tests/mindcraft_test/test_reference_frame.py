from unittest import TestCase


class TestReferenceFrame(TestCase):
    def test_import(self, verbose=False):
        from mindcraft import ReferenceFrame
        if verbose:
            print(ReferenceFrame)
