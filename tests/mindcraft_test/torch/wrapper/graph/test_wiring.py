from unittest import TestCase


class TestWiring(TestCase):
    def test_import_self_attention_map(self, verbose=False):
        from mindcraft.torch.wrapper.graph import Wiring
        if verbose:
            print(Wiring)
