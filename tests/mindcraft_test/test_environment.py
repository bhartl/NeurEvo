from unittest import TestCase


class TestEnv(TestCase):
    def test_import(self, verbose=False):
        from mindcraft import Env
        if verbose:
            print(Env)
