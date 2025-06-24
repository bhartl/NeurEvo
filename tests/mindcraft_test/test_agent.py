from unittest import TestCase


class TestAgent(TestCase):
    def test_import(self, verbose=False):
        from mindcraft import Agent
        if verbose:
            print(Agent)
