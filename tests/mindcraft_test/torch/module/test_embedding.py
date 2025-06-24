from unittest import TestCase


class TestStateEmbedding(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import StateEmbedding
        if verbose:
            print(StateEmbedding)


class TestGRNEmbedding(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import GRNEmbedding
        if verbose:
            print(GRNEmbedding)


class TestSensoryEmbedding(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import SensoryEmbedding
        if verbose:
            print(SensoryEmbedding)
