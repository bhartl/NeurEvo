from unittest import TestCase
import numpy as np
import itertools as it

import torch.linalg


class TestSetTransformer(TestCase):
    """ Testcases for SetTransformer module - mostly permutation invariance, etc.

        (c) B. Hartl 2021
    """
    def test_reset(self, verbose=False):
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import Projection, LinearP

        batch_size = 1
        n_features = 6
        n_channels = 4

        key_embed = 3
        val_embed = 3
        query_embed = 3
        val_size = 1
        context_size = 9

        channels_first = False
        np.random.seed(18648)

        if channels_first:
            x = (Tensor(np.random.rand(batch_size, n_channels, n_features)) - 0.5)
        else:
            x = (Tensor(np.random.rand(batch_size, n_features, n_channels)) - 0.5)

        x *= 1e-3

        for a, n, p, s in it.product(['Tanh', 'ReLU'],
                                     ['Identity', 'LayerNorm'],
                                     [None, LinearP, ],
                                     ['LSTM', 'RNN', 'GRU', None],  # 'NCP'],
                                     ):

            if p is None:
                if s is None:
                    continue

                s = Recurrent(input_size=n_channels, output_size=key_embed, layer_type=s, num_layers=2, is_nested=True) if s else None
                v = LinearP(input_size=n_channels, projection_size=val_embed, is_nested=True)
            else:
                Recurrent(input_size=key_embed, layer_type=s, is_nested=True) if s else None
                p = p(input_size=n_channels, projection_size=key_embed, is_nested=True, )
                v = LinearP(input_size=n_channels, projection_size=val_embed, is_nested=True, )

            key_embedding = SensoryEmbedding(projection=p, sensor=s, is_nested=True)

            val_embedding = SensoryEmbedding(projection=v, is_nested=True)

            sensor = SetTransformer(seq_len=n_features,
                                    input_size=n_channels,
                                    key_embed=key_embedding,
                                    val_embed=val_embedding,
                                    qry_size=query_embed,
                                    context_size=context_size,
                                    val_size=val_size,
                                    qkv_bias=True,
                                    activation=a,
                                    norm_layer=n,
                                    channels_first=channels_first,
                                    disable_pos_embed=True,
                                    retain_grad=False,
                                    )

            if verbose: print(f'set transformer model: {get_n_params(sensor)} parameters, setting: ', a, n, s)
            sensor.trainable = False
            p = sensor.serialize_parameters()
            sensor.deserialize_parameters(Tensor(np.random.rand(*p.shape)))

            f = sensor(x, )
            if verbose: print(f"froward input (shape {x.shape}) to latent output (shape {f.shape})")

            # compare reloaded parameters
            sensor2 = SetTransformer.make(sensor.to_dict())

            params_sensor = sensor.serialize_parameters()
            params_sensor2 = sensor2.serialize_parameters()
            self.assertTrue(np.array_equal(params_sensor, params_sensor2))

            # compare forward pass
            f2 = sensor2(x, )
            self.assertTrue(np.allclose(f, f2))

            if verbose: print(f"permute features (reverse order of features)")
            sensor.reset()
            f_reset = sensor(x, )
            for i, (fi, fi_p) in enumerate(zip(f.detach().numpy(), f_reset.detach().numpy())):
                assert np.array_equal(fi,
                                      fi_p), f"Deviation in the following arrays detected\n" \
                                             f"original: {fi.tolist()}\n" \
                                             f"permuted: {fi_p.tolist()}"

    def test_forward(self, verbose=False):
        import torch
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import LinearP

        batch_size = 32
        n_features = 24
        n_channels = 16

        key_embed = 22
        val_embed = 26
        query_embed = 74
        val_size = 12
        context_size = 17

        action_dim = 41

        channels_first = False
        np.random.seed(18648)

        if channels_first:
            x = (Tensor(np.random.rand(batch_size, n_channels, n_features)) - 0.5)
        else:
            x = (Tensor(np.random.rand(batch_size, n_features, n_channels)) - 0.5)

        x *= 1e3
        z = Tensor(np.random.rand(batch_size, action_dim, ))

        for a, n, s in it.product(['Tanh', 'ReLU'],
                                  ['Identity', 'LayerNorm'],
                                  ['LSTM', 'RNN', 'GRU',],
                                  ):

            key_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels + action_dim, projection_size=key_embed, ),
                                             sensor=Recurrent(input_size=key_embed),
                                             )

            val_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=val_embed, ),
                                             sensor=Recurrent(input_size=val_embed),
                                             )

            sensor = SetTransformer(seq_len=n_features,
                                    input_size=n_channels,
                                    key_embed=key_embedding,
                                    val_embed=val_embedding,
                                    qry_size=query_embed,
                                    context_size=context_size,
                                    val_size=val_size,
                                    qkv_bias=True,
                                    activation=a,
                                    norm_layer=n,
                                    channels_first=channels_first,
                                    disable_pos_embed=True,
                                    )

            if verbose: print(f'set transformer model: {get_n_params(sensor)} parameters, setting: ', a, n, s)
            sensor.trainable = False
            p = sensor.serialize_parameters()
            sensor.deserialize_parameters(Tensor(np.random.rand(*p.shape)))

            sensor_repr = sensor.to_dict()
            sensor_restored = SetTransformer.make(sensor_repr)
            if verbose: print(f"check restored parameters")
            for k, v in sensor.state_dict().items():
                v_restored = sensor_restored.state_dict()[k]
                self.assertTrue(torch.equal(v, v_restored))

            x_original, z_original = x, z
            if verbose: print(f"froward input (shape {x.shape}) to latent output (shape {f.shape})")
            f = sensor(x, z)
            self.assertTrue(torch.equal(z, z_original))
            self.assertTrue(torch.equal(x, x_original))

    def test_restore(self, verbose=False):
        import torch
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import LinearP

        batch_size = 1
        n_features = 24
        n_channels = 12

        key_embed = 22
        val_embed = 26
        query_embed = 74
        val_size = 12
        context_size = 17

        channels_first = False
        np.random.seed(18648)

        if channels_first:
            x = (Tensor(np.random.rand(batch_size, n_channels, n_features)) - 0.5)
        else:
            x = (Tensor(np.random.rand(batch_size, n_features, n_channels)) - 0.5)

        x *= 1e3

        for a, n, s in it.product(['Tanh', 'ReLU'],
                                  ['Identity', 'LayerNorm'],
                                  ['LSTM', 'RNN', 'GRU', ],
                                  ):

            key_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=key_embed, ),
                                             sensor=Recurrent(input_size=key_embed, layer_type=s),
                                             )

            val_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=val_embed, ),
                                             )

            sensor = SetTransformer(seq_len=n_features, input_size=n_channels,
                                    key_embed=key_embedding, val_embed=val_embedding,
                                    qry_size=query_embed,
                                    context_size=context_size, val_size=val_size,
                                    qkv_bias=True, activation=a, norm_layer=n,
                                    channels_first=channels_first, disable_pos_embed=True,
                                    )

            if verbose: print(f'set transformer model: {get_n_params(sensor)} parameters, setting: ', a, n, s)
            sensor.trainable = False
            p = sensor.serialize_parameters()
            sensor.deserialize_parameters(Tensor(np.random.rand(*p.shape)))

            sensor_restored = SetTransformer.make(sensor.to_dict())
            if verbose: print(f"check restored parameters")
            for k, v in sensor.state_dict().items():
                v_restored = sensor_restored.state_dict()[k]
                self.assertTrue(torch.equal(v, v_restored))

            f = sensor(x, )
            if verbose: print(f"froward input (shape {x.shape}) to latent output (shape {f.shape})")

            f_restored = sensor_restored(x, )
            for fi in f_restored.detach().numpy():
                if verbose: print(fi.tolist())

            self.assertTrue(np.array_equal(f.detach().numpy(), f_restored.detach().numpy()))

    def test_batch_permute(self, verbose=False):
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import LinearP

        batch_size = 33
        n_features = 12
        n_channels = 3

        key_embed = 10
        val_embed = 11
        query_embed = 12
        val_size = 3
        context_size = 8

        np.random.seed(18648)

        key_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=key_embed, ),
                                         sensor=Recurrent(input_size=key_embed, layer_type='LSTM', stateful=True),
                                         )

        val_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=val_embed, ),
                                         )

        sensor = SetTransformer(seq_len=n_features, input_size=n_channels,
                                key_embed=key_embedding, val_embed=val_embedding,
                                qry_size=query_embed,
                                context_size=context_size, val_size=val_size,
                                qkv_bias=True, activation='Tanh', norm_layer='Identity',
                                channels_first=False, disable_pos_embed=True,
                                )

        if verbose: print(f'set transformer model: {get_n_params(sensor)} parameters')
        p = sensor.serialize_parameters()
        sensor.deserialize_parameters(Tensor(np.random.rand(*p.shape)) * 1.)
        sensor.trainable = False

        for t in range(1, 10, 2):
            x = (Tensor(np.random.rand(batch_size, n_features, n_channels)) - 0.5)
            x *= 1e0

            sensor.reset()
            k, v = [], []
            for _ in range(t):
                ki, vi = sensor.embed(x, )
                k.append(ki), v.append(vi)

            if verbose: print()
            if verbose: print(f"key-embedding (shape {k[-1].shape}) and value-embedding (shape )after {t} time steps")
            if verbose: print(f"permute batches (reverse order of batches)")
            sensor.reset()
            permute = np.arange(0, len(x))
            np.random.shuffle(permute)

            for i in range(t):
                k_p, v_p = sensor.embed(x[permute], )

                ki, vi = k[i], v[i]

                for i, (ki_o, ki_p) in enumerate(zip(ki.detach().numpy()[permute], k_p.detach().numpy())):
                    self.assertTrue(np.allclose(ki_o, ki_p), f"Deviation in BATCH {i} of the following key embeddings\noriginal: {ki.tolist()}\npermuted: {ki_p.tolist()}")

                for i, (vi_o, vi_p) in enumerate(zip(vi.detach().numpy()[permute], v_p.detach().numpy())):
                    self.assertTrue(np.allclose(vi_o, vi_p), f"Deviation in BATCH {i} of the following key embeddings\noriginal: {ki.tolist()}\npermuted: {ki_p.tolist()}")

    def test_feature_permutation_invariance(self, verbose=False, accuracy=5):
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import LinearP

        batch_size = 1
        n_features = 4
        n_channels = 3

        key_embed = 3
        val_embed = 2
        query_embed = 2
        val_size = 1
        context_size = 2

        np.random.seed(18648)

        for a, n, s in it.product(['Tanh', ],
                                  ['Identity', ],
                                  ['LSTM', 'RNN'],
                                  ):

            x = (Tensor(np.random.rand(batch_size, n_features, n_channels)) - 0.5)

            x *= 1e0

            key_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=key_embed, ),
                                             sensor=Recurrent(input_size=key_embed, layer_type='LSTM', stateful=True),
                                             )

            val_embedding = SensoryEmbedding(projection=LinearP(input_size=n_channels, projection_size=val_embed, ),
                                             )

            sensor = SetTransformer(seq_len=n_features, input_size=n_channels,
                                    key_embed=key_embedding, val_embed=val_embedding,
                                    qry_size=query_embed,
                                    context_size=context_size, val_size=val_size,
                                    qkv_bias=True, activation='Tanh', norm_layer='Identity',
                                    channels_first=False, disable_pos_embed=True,
                                    )

            if verbose: print(f'set transformer model: {get_n_params(sensor)} parameters, setting: ', a, n, s)
            p = sensor.serialize_parameters()
            sensor.deserialize_parameters(Tensor(np.random.rand(*p.shape)))
            sensor.trainable = False

            for t in range(1, 20, 5):
                sensor.reset()
                f = None
                for _ in range(t):
                    f = sensor(x, )

                if verbose: print(f"latent encoding (shape {f.shape}) after {t} time-steps")
                permute = np.arange(0, x.shape[1])
                np.random.shuffle(permute)
                if verbose: print(f"permute features", permute)

                sensor.reset()
                f_p = None
                for _ in range(t):
                    f_p = sensor(x[:, permute, :], )

                for i, (fi, fi_p) in enumerate(zip(f.detach().numpy(), f_p.detach().numpy())):
                    fi = np.round(fi, accuracy)
                    fi_p = np.round(fi_p, accuracy)
                    self.assertTrue(np.array_equal(fi, fi_p), f"Deviation in BATCH {i} of the following key embeddings\noriginal: {fi.tolist()}\npermuted: {fi_p.tolist()}")
