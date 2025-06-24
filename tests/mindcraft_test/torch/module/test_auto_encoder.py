from unittest import TestCase, skip


class TestAutoEncoder(TestCase):
    def test_import(self, verbose=False):
        from mindcraft.torch.module import AutoEncoder
        if verbose:
            print(AutoEncoder)

    @skip
    def test_encoder_decoder(self, verbose=False):
        from mindcraft.torch.module import Conv
        from mindcraft.torch.module import ConvT
        from mindcraft.torch.module import AutoEncoder

        encoder = Conv(
            input_size=1,
            input_dim=2,
            kernel_size=(3, 3, 3, 5),
            filters=(32, 64, 64, 2),
            strides=(1, 2, 2, 1),
            activation=('ReLU', 'ReLU', 'ReLU', None),
            flatten=dict(cls="FeedForward", input_size=1600, output_size=2),
        )

        decoder = ConvT(
            input_size=2,
            input_dim=2,
            filters=[64, 64, 32, 1],
            kernel_size=[4, 4, 4, 4],
            strides=[1, 3, 3, 2],
            activation=['ReLU', 'ReLU', 'ReLU', 'Sigmoid'],
        )

        auto_encoder = AutoEncoder(encoder=encoder, decoder=decoder)

        from torch import randn
        x_0 = randn(10, 1, 28, 28)  # create 28 x 28 arrays with 1 channel each
        if verbose:
            print("{0:<22s}".format("encoder input:"), x_0.shape)
            x = x_0
            for cnn in auto_encoder.encoder.cnn:
                x = cnn(x)
                print("- {0:<20s} output:".format(str(cnn.__class__.__name__)), x.shape)

        x = x_0
        z = auto_encoder.encode(x)
        self.assertTrue(len(z.shape) == 2)
        self.assertTrue(z.shape[0] == x.shape[0])
        self.assertTrue(z.shape[1] == 2)

        y_decode = auto_encoder.decode(z)
        y_through = auto_encoder(x)

        from mindcraft.torch.util import tensor_to_numpy
        from numpy import array_equal
        self.assertTrue(array_equal(tensor_to_numpy(y_through), tensor_to_numpy(y_decode)))
