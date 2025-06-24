from unittest import TestCase


class TestSelfAttention(TestCase):
    def test_import_self_attention_map(self, verbose=False):
        from mindcraft.torch.wrapper.np import SelfAttentionMap
        if verbose:
            print(SelfAttentionMap)

    def test_import_self_attention(self, verbose=False):
        from mindcraft.torch.wrapper.np import SelfAttention
        if verbose:
            print(SelfAttention)
