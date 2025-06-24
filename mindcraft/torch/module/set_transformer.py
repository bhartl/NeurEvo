from torch import zeros, concatenate
from torch.nn import Module, Linear, Identity, Parameter
from torch.nn import Softmax
from torch.nn import LayerNorm
from math import sqrt
from mindcraft.torch.module import Patchwork, SensoryEmbedding
from mindcraft.torch.module import SensoryEmbedding as Embedding
from mindcraft.torch.module import PatchP
from mindcraft.torch.util import get_torch_layer, get_activation_function, layer_type_repr
from functools import partial
from typing import Union, Optional


class SetTransformer(Patchwork):
    """ A permutation invariant SetTransformer, that applies a single QK-V attention layer with fixed Q-Matrix
        on an input sequence to form a permutation invariant context.

        (c) B. Hartl 2021
    """
    
    REPR_FIELDS = ("input_size", "seq_len", "channels_first",
                   "key_embed", "val_embed",
                   "qry_size", "context_size", "val_size", "activation",
                   "disable_pos_embed", "qkv_bias", "norm_layer", "head", "patch_embedding",
                   *Patchwork.REPR_FIELDS
                   )

    def __init__(self,
                 input_size=None,
                 seq_len=None,
                 channels_first: bool = True,
                 key_embed: Union[str, dict, Embedding] = None,
                 val_embed: Union[str, dict, Embedding] = None,
                 norm_layer=Identity,
                 qry_size=8,
                 context_size=32,
                 val_size=1,
                 activation=Softmax,
                 disable_pos_embed=False,
                 qkv_bias=True,
                 head=None,
                 patch_embedding=False,
                 **patchwork_kwargs
                 ):
        """ Constructs a SensoryTransformer instance

        :param input_size: (int) number of input channels
        :param seq_len: (int) total number of input features, i.e., the length of the input sequence.
        :param key_embed_size: (int) dimension (cols) for key sensory-embedding (function of input + feedback)
        :param val_embed_size: (int) dimension (cols) for value sensory-embedding (function of input)
        :param qry_size: (int) embedding size (cols) of query matrix (fixed)
        :param context_size: (int) global latent space row-dim (rows of query matrix) of attention, defaults to 32.
        :param val_size: (int) latent space col-dim (cols of value projection matrix), defaults to 1.
        :param disable_pos_embed: (bool) flag to disable positional embedding of patches.
        :param qkv_bias: (bool) enable bias for qkv if True
        :param activation: (callable) elementwise activation `sigma`-function layer for `sigma(Q K.T / sqrt(d_q)) evaluation
        :param norm_layer: (callable) Optional normalization layer, e.g. 'LayerNorm' or 'GroupNorm', defaults to 'Identity'.
        :param sensor_num_layers: (int) Number of Sensory layers after token embedding, defaults to 1.
        :param sensor_type: (str) Choice of ('RNN', 'LSTM', 'GRU', 'NCP'), defaults to 'RNN'.
        :param sensor_kwargs: (Optional dict) Keywords to be passed to Sensory layer.
        :param key_proj_type: Projection layer type (str representation or torch.nn.Layer) for key-embedding,
                              defaults to 'Linear'.
        :param key_proj_kwargs: (Optional) keyword argument dictionary for the specified `key_proj_type`,
                                defaults to None.
        :param val_proj_type: Projection layer type (str representation or torch.nn.Layer) for value-embedding,
                              defaults to 'Linear'.
        :param val_proj_kwargs: (Optional) keyword argument dictionary for the specified `val_proj_type`,
                                defaults to None.
        :param head: Optional MLP layer for postprocessing the normalized context, defaults to None.
        :param patch_embedding: (bool) Flag to use PatchP projection layer for 2D input, defaults to False.
        """
        Module.__init__(self)
        
        # nn parameters
        self.key_embed = key_embed
        self.val_embed = val_embed
        self.pos_embed = None
        self.key_matrix = None
        self.val_matrix = None
        self.qry_matrix = None
        self.input_norm = None
        self.context_norm = None
        self.qry_norm = None
        self.sigma = None

        self.seq_len = seq_len
        self.norm_layer = norm_layer

        # network architecture
        self.qry_size = qry_size
        self.context_size = context_size
        self.val_size = val_size

        self.qkv_bias = qkv_bias
        self.disable_pos_embed = disable_pos_embed
        self.activation = activation

        # positional embedding
        self.input_size = input_size
        self.seq_len = seq_len

        # functional parameters
        self.channels_first = channels_first
        self.head = head

        # helpers
        self._add_batch_dim = False
        self.patch_embedding = patch_embedding

        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self):
        # key embedding
        if self.key_embed is None:
            self.key_embed = Embedding()
        self.key_embed = Embedding.make(self.key_embed)
        if self.key_embed.is_sequence_module:
            assert self.key_embed.sensor.rnn.batch_first
        if self.input_size is None:
            assert self.key_embed is not None, "Define either key_embed or input_size."
            self.input_size = self.key_embed.input_size

        # value embedding
        if self.val_embed is None:
            self.val_embed = Embedding()
        self.val_embed = Embedding.make(self.val_embed)
        if self.val_embed.is_sequence_module:
            assert self.val_embed.sensor.rnn.batch_first

        key_embed_size = self.key_embed.embed_size or self.input_size
        val_embed_size = self.val_embed.embed_size or self.input_size
        self.key_matrix = Linear(key_embed_size, self.qry_size, bias=self.qkv_bias)
        self.val_matrix = Linear(val_embed_size, self.val_size, bias=self.qkv_bias)
        self.qry_matrix = Linear(self.qry_size, self.context_size, bias=self.qkv_bias)
        self.qry_norm = sqrt(self.qry_size)

        # norm layer
        norm_layer = get_torch_layer(self.norm_layer or partial(LayerNorm, eps=1e-6))
        self.input_norm = norm_layer(self.input_size) if norm_layer is LayerNorm else norm_layer()
        self.context_norm = norm_layer(self.val_size) if norm_layer is LayerNorm else norm_layer()

        # activation layer
        activation = get_activation_function(self.activation or Softmax)

        self.sigma = activation()
        self.qry_norm = 1. / sqrt(self.qry_size)

        if not self.disable_pos_embed:
            assert self.seq_len, "missing `input_len`, positional embedding only with fixed number of features."
            if self.is_sensor_embedding:
                embed_dim = self.key_embed.embed_size
            else:
                embed_dim = self.input_size

            self.pos_embed = Parameter(zeros(1, self.seq_len, embed_dim))

        if self.head is not None:
            self.head = Patchwork.make(self.head)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        dict_repr['activation'] = layer_type_repr(self.activation)
        dict_repr['norm_layer'] = layer_type_repr(self.norm_layer)

        if dict_repr.get('key_embed', None):
            dict_repr['key_embed'].pop('serialized', None)
            dict_repr['key_embed'].pop('recover_indices', None)
            dict_repr['key_embed'].pop('serialize_mask', None)

        if dict_repr.get('val_embed', None):
            dict_repr['val_embed'].pop('serialized', None)
            dict_repr['val_embed'].pop('recover_indices', None)
            dict_repr['val_embed'].pop('serialize_mask', None)

        return dict_repr

    def reset(self):
        self.key_embed.reset()
        self.val_embed.reset()

    def forward(self, x, *args):
        # input code normalization, defaults to Identity
        x = self.input_norm(x)

        # embedding
        key_embedding, val_embedding = self.embed(x, *args)

        # Matrix multiplication of `Q @ K_T / sqrt(q_dim)`:
        Q_KT = (self.qry_matrix(self.key_matrix(key_embedding)) * self.qry_norm).transpose(-2, -1)

        # Matrix multiplication of `QK_T @ V`:
        context = self.sigma(Q_KT) @ self.val_matrix(val_embedding)

        # output code normalization, defaults to Identity
        context = self.context_norm(context)
        context = context.reshape(*context.shape[:-2], -1)  # merge val-dim
        if self.head is not None:
            if self.head.is_sequence_module:
                context = context.unsqueeze(1)
            context = self.head(context)
            if self.head.is_sequence_module:
                context = context.squeeze(1)

        if self._add_batch_dim:
            context = context.squeeze(0)

        return context

    def embed(self, x, *args):
        if self.patch_embedding:
            return self.embed_patch(x, *args)

        return self.embed_seq(x, *args)

    def embed_seq(self, x, *args):
        """ Embed feature vector into key and value embedding

        :param x: Feature Vector of dimension (B, N, C)
        :return: key_embedding, val_embedding of dims (B, N, key_embed_size) and (B, N, val_embed_size)
        """
        # prepare shape
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # introduce batch dim
            self._add_batch_dim = True

        assert len(x.shape) == 3, "Assuming (B, C, N) or (B, N, C) shape a.t.m."
        if self.channels_first:  # shape: B, C, N
            x = x.permute(0, 2, 1)  # to ->: B, N, C
        # if z is not None and len(z.shape) == 1:
        #     z = z.unsqueeze(0).repeat(x.shape[0], 1)  # introduce batch-size, B, to foldback z
        # if z is not None and len(z.shape) == 2:  # assume shape
        #     assert z.shape[0] == x.shape[0], "Batch-size dim-0 of x and z doesn't match."  # assume same batch-size
        #     # if z.shape[1] == x.shape[1]:  # same number of features
        #     #     z = z.unsqueeze(2)        # introduce channels, B, N -> B, N, (C_z = 1)
        #     # else:                         # B, C_z -> B, N, C_z
        #     z = z.unsqueeze(1).repeat(1, x.shape[1], 1)

        x = self.embed_pos(x)
        B, N = x.shape[0], x.shape[1]
        x = x.reshape(B * N, 1, -1)  # squeeze B, N, C_z -> B x N, 1, C_z with time-series len = 1

        _args = []
        for xi in args:
            if xi.ndim == 2 and xi.shape[0] == B:
                # extend batch-dim by N: (B, C) -> (B * N, C)
                xi = xi.view(B, 1, -1).expand(-1, N, -1)
                xi = xi.reshape(B * N, 1, -1)  # squeeze B, N, C_z -> B x N, 1, C_z with time-series len = 1

            else:
                if xi.ndim == 2 and len(xi) == 1:
                    xi = xi[0]

                xi = xi.view(1, 1, -1).expand(N, -1, -1)

            _args.append(xi)

        kx = concatenate([x, *_args], dim=-1 if not self.channels_first else -2)  # concatenate along feature dim
        k = self.key_embed(kx)
        v = self.val_embed(x)
        return k.reshape(B, N, -1), v.reshape(B, N, -1)

    def embed_patch(self, x, *args):
        """ PatchEmbed 2D feature vector (image or alike) into key and value embedding

        :param x: Feature Vector of dimension (B, Nx, Ny, C)
        :return: key_embedding, val_embedding of dims (B, N_feat, key_embed_size) and (B, N_feat, val_embed_size)
        """
        # prepare shape
        if len(x.shape) == 3:   # (Nx, Ny, C) or (C, Nx, Ny)
            x = x.unsqueeze(0)  # introduce batch dim
            self._add_batch_dim = True

        assert len(x.shape) == 4, "Assuming shape (B, Nx, Ny, C) or (B, C, Nx, Ny)."
        if not self.channels_first:  # shape: B, C, Nx, Ny
            x = x.permute(0, 3, 1, 2)  # to ->: B, Nx, Ny, C
        # if len(z.shape) == 1:
        #     z = z.unsqueeze(0).repeat(x.shape[0], 1)  # introduce batch-size, B, to foldback z
        # if len(z.shape) == 2:
        #     assert z.shape[0] == x.shape[0], "Batch-size dim-0 of x and z doesn't match."  # assume same batch-size
        #     # if z.shape[1] == x.shape[1]:  # same number of features
        #     #     z = z.unsqueeze(2)        # introduce channels, B, N -> B, N, (C_z = 1)
        #     # else:                         # B, C_z -> B, N, C_z
        #     z = z.unsqueeze(1).repeat(1, self.n_features, 1)

        B = x.shape[0]  # batch size

        k = self.key_embed(self.merge_tensors(x, *args))
        v = self.val_embed(x)

        if len(k.shape) == 5:
            k = k.view(B, self.key_embed.projection.num_patches, -1)

        if len(v.shape) == 5:
            v = v.view(B, self.val_embed.projection.num_patches, -1)

        k = self.embed_pos(k)
        return k, v

    def merge_tensors(self, x, *args):
        n_b, n_t, *n_i = x.shape  # observation shape, but features
        x_dim = len(x.shape)

        x = [x]
        for xi in args:
            assert xi.ndim == 1, f"attr {xi} must be 1D"

            if x_dim == 3:  # (B * N , 1, C)
                # add batch_dim and expand to match n_b
                xi = xi.view(1, 1, -1).expand(n_b, -1, -1)


            else:  # Expand the 1D tensor to match the Nx, Ny, Nz, ... shape
                xi = xi.view(-1, *([1]*len(n_i))).expand(1, -1, *n_i)

            x.append(xi)

        x = concatenate([x, *args], dim=1 if x_dim != 3 else -1)  # concatenate along feature dim
        return x

    def embed_pos(self, x):
        if self.pos_embed is not None:
            x = x + self.pos_embed
            # TODO: Interpolate
        return x

    @property
    def is_sensor_embedding(self):
        return isinstance(self.key_embed, SensoryEmbedding)

    @property
    def is_sequence_module(self):
        is_sequence = False
        try:
            is_sequence |= self.key_embed.is_sequence_module
        except TypeError:
            pass

        try:
            is_sequence |= self.val_embed.is_sequence_module
        except TypeError:
            pass

        try:
            is_sequence |= self.head.is_sequence_module
        except (TypeError, AttributeError):
            pass

        return is_sequence

    @property
    def states(self):
        if not self.is_sequence_module:
            return None

        elif not self.val_embed.is_sequence_module:
            return self.key_embed.states

        elif not self.key_embed.is_sequence_module:
            return self.val_embed.states

        key_embed_states = self.key_embed.states
        val_embed_states = self.val_embed.states

        if key_embed_states is None and val_embed_states is None:
            # both initialized
            return None

        return *key_embed_states, *val_embed_states

    @states.setter
    def states(self, value):
        assert self.is_sequence_module
        val_offset = 0
        if self.key_embed.is_sequence_module:
            val_offset = len(self.key_embed.states)
            self.key_embed.states = value[:val_offset]

        if self.val_embed.is_sequence_module:
            self.val_embed.states = value[val_offset:]

    @property
    def num_layers(self) -> Union[int, tuple]:
        if not self.is_sequence_module:
            return 0
        elif not self.val_embed.is_sequence_module:
            return self.key_embed.num_layers
        elif not self.key_embed.is_sequence_module:
            return self.val_embed.num_layers
        return *self.num_val_layers, *self.num_val_layers

    @property
    def num_key_layers(self) -> tuple:
        if not self.key_embed.is_sequence_module:
            return ()
        num_key_layers = [self.key_embed.num_layers]
        if self.key_embed.states:
            num_key_layers = num_key_layers * len(self.key_embed.states)
        return tuple(num_key_layers)

    @property
    def hidden_size(self) -> Union[int, tuple]:
        if not self.is_sequence_module:
            return 0
        elif not self.val_embed.is_sequence_module:
            return self.key_embed.hidden_size
        elif not self.key_embed.is_sequence_module:
            return self.val_embed.hidden_size
        return *self.num_val_layers, *self.num_val_layers

    @property
    def num_val_layers(self) -> tuple:
        if not self.val_embed.is_sequence_module:
            return ()
        num_val_layers = [self.val_embed.num_layers]
        if self.val_embed.states:
            num_val_layers = num_val_layers * len(self.val_embed.states)
        return tuple(num_val_layers)

    @property
    def hidden_key_size(self) -> tuple:
        if not self.key_embed.is_sequence_module:
            return ()
        hidden_key_size = [self.key_embed.hidden_size]
        if self.key_embed.states:
            hidden_key_size = hidden_key_size * len(self.key_embed.states)
        return tuple(hidden_key_size)

    @property
    def hidden_val_size(self) -> tuple:
        if not self.val_embed.is_sequence_module:
            return ()
        hidden_val_size = [self.val_embed.hidden_size]
        if self.val_embed.states:
            hidden_val_size = hidden_val_size * len(self.val_embed.states)
        return tuple(hidden_val_size)

    def aggregates(self, dim):
        return dim == 1

    @property
    def output_size(self):
        if self.head is None:
            return self.context_size

        return self.head.output_size