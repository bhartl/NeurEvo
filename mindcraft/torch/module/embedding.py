from abc import ABC
import torch
from torch import Tensor, zeros, normal, index_select, randn, tanh
from torch import abs as torch_abs
from torch.nn import Parameter
from mindcraft.torch.layer import GRN
from torch.nn import Module
from typing import Union, Optional
from mindcraft.torch.module import Patchwork
from mindcraft.torch.module import Projection, Recurrent


class Embedding(Patchwork, ABC):
    REPR_FIELDS = tuple(r for r in Patchwork.REPR_FIELDS)

    def __init__(self,
                 **patchwork_kwargs
                 ):

        patchwork_kwargs["omit_default"] = patchwork_kwargs.get("omit_default", True)
        Patchwork.__init__(self, **patchwork_kwargs)

    @property
    def input_size(self):
        raise NotImplementedError()

    @property
    def embed_size(self) -> Optional[int]:
        raise NotImplementedError()

    @property
    def output_size(self) -> Optional[int]:
        raise NotImplementedError()


class StateEmbedding(Embedding):
    """ A module providing a tensor comprising state parameters
        for a number of different possible states, e.g., assumed
        in the initial seed of a developmental phase in an NCA.

        Thus, the states of a state module can be seen as different
        gene expressions for certain experiments."""
    REPR_FIELDS = ("state_size", "num_states", "randomize", *Embedding.REPR_FIELDS)

    def __init__(self,
                 state_size: int,
                 num_states: int = 1,
                 randomize: bool = True,
                 **patchwork_kwargs
                 ):

        self.state_size = state_size
        self.num_states = num_states
        self.randomize = randomize

        # torch Parameters
        self.state = None
        Embedding.__init__(self, **patchwork_kwargs)

    def _build(self):
        if self.randomize:
            state = normal(0., 1., (self.num_states, self.state_size))
        else:
            state = zeros((self.num_states, self.state_size))

        self.state = Parameter(state)

    def to(self, device) -> 'StateEmbedding':
        Embedding.to(self, device)
        self.state = Parameter(self.state.to(device))
        return self

    def forward(self, x: Tensor, *args) -> Tensor:
        """ Returns states according to state-indices specified in `x`

        For example, an input tensor `x = [1, 0]` will return the tensor `[state[1], state[0]]`.

        :param x: Integer array/tensor addressing different states (dim 1)
        :return: Tensor of states addressed by `x`.
        """
        return index_select(self.state, dim=0, index=x)

    @property
    def input_size(self):
        return 1

    @property
    def embed_size(self) -> Optional[int]:
        return self.state_size

    @property
    def output_size(self) -> Optional[int]:
        return self.state_size


class GRNEmbedding(Embedding):
    REPR_FIELDS = ("num_genes", "tau_1", "tau_2", "num_iterations", *Embedding.REPR_FIELDS)

    def __init__(self,
                 num_genes: int,
                 tau_1: float = 1.,
                 tau_2: float = 0.2,
                 num_iterations: int = 6,
                 **patchwork_kwargs
                 ):

        self.num_genes = num_genes
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.num_iterations = num_iterations

        # torch Parameters
        self.grn = None
        Embedding.__init__(self, **patchwork_kwargs)

    def _build(self):
        self.grn = GRN(num_genes=self.num_genes, )

    def to(self, device) -> 'GRNEmbedding':
        Embedding.to(self, device)
        return self

    def forward(self, x: Tensor, *args) -> Tensor:
        """ Returns the regulated `x` after `num_iterations` of applying the GRN

        :param x: Input tensor.
        :return: Regulated gene expression.

        .. math::

        x_{t+1} = (1 - \\tau_2) \\times x_{t} + \\tau_1 \\sigma(W \\times x_{t})
        """
        for _ in range(self.num_iterations):
            x = self.grn(x)

        return x

    @property
    def input_size(self):
        return self.num_genes

    @property
    def embed_size(self) -> Optional[int]:
        return self.num_genes

    @property
    def output_size(self) -> Optional[int]:
        return self.embed_size


class ConcatEmbedding(Embedding):
    """ Embedding Module which concatenates (flattens) the input of the module along dim 1. """
    REPR_FIELDS = ("input_size", "num_channels", *Embedding.REPR_FIELDS)

    def __init__(self,
                 input_size: int,
                 num_channels: int,
                 **patchwork_kwargs
                 ):
        """ Constructs a ConcatEmbedding Instance """
        self._input_size = input_size
        self.num_channels = num_channels
        Embedding.__init__(self, **patchwork_kwargs)

    def _build(self):
        pass

    def to(self, device) -> 'ConcatEmbedding':
        Embedding.to(self, device)
        return self

    def forward(self, x: Tensor, *args) -> Tensor:
        """ Returns the concatenation of the input `x` in dim 1 """
        x = super().forward(x, *args)
        while len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        return x

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def embed_size(self) -> int:
        return self.input_size * self.num_channels

    @property
    def output_size(self) -> int:
        return self.embed_size

    def to_dict(self):
        dict_repr = Embedding.to_dict(self)
        dict_repr['serialized'] = None
        return dict_repr


class SensoryEmbedding(Embedding):
    """ Embedding Module of multichannel feature input of arbitrary length.

        (c) B. Hartl 2021
    """

    REPR_FIELDS = ("projection", "sensor", *Embedding.REPR_FIELDS)

    def __init__(self,
                 projection: Optional[Union[str, dict, Projection]] = None,
                 sensor: Optional[Union[str, dict, Recurrent]] = None,
                 **kwargs,
                 ):
        """ Constructs an Embedding Instance

        :param projection:
        :param sensor:
        """
        Module.__init__(self)
        self.projection = projection
        self.sensor = sensor
        Embedding.__init__(self, **kwargs)

    def _build(self):
        if self.projection is not None:
            self.projection = Projection.make(self.projection)

        if self.projection is not None:
            self.projection.is_nested = True

        if self.sensor is not None:
            self.sensor = Recurrent.make(self.sensor)

        if self.sensor is not None:
            self.sensor.is_nested = True

    def to_dict(self):
        dict_repr = Embedding.to_dict(self)

        if 'projection' in dict_repr:
            dict_repr['projection'].pop('serialized', None)
            dict_repr['projection'].pop('recover_indices', None)
            dict_repr['projection'].pop('serialize_mask', None)

        if 'sensor' in dict_repr:
            dict_repr['sensor'].pop('serialized', None)
            dict_repr['sensor'].pop('recover_indices', None)
            dict_repr['sensor'].pop('serialize_mask', None)

        return dict_repr

    @property
    def input_size(self):
        if self.projection is None:
            if self.sensor is None:
                return 0
            return self.sensor.input_size
        return self.projection.input_size

    @property
    def projection_size(self):
        if self.projection is None:
            if self.sensor is None:
                return 0
            return self.sensor.output_size
        return self.projection.projection_size

    @property
    def embed_size(self) -> Optional[int]:
        if self.sensor is None:
            return self.projection.projection_size

        return self.sensor.output_size

    @property
    def output_size(self) -> Optional[int]:
        return self.embed_size

    def forward(self, x, *args):
        """ Transforms an input, `x`, into an embedding.

        - First, a linear projection of the input, `x`, is performed, transforming the input of size `input_size`
          to a `projection_size`
        - Then, a SensoryNeuron instance is applied - **if defined** -
          to the input-projection to form an embedding.

        :param x: (torch.Tensor) input tensor to the neural network of dim `input_size` of the non-Identity project,
                  else of the `input_size` of the sensor.
        :returns: The feature-wise embedding of the input.
        """
        if self.projection is None:
            return self.sensor(x, *args)

        x = self.projection(x, *args)
        is_2d = len(x.shape) == 4  # (BS, FEATURES, X, Y) or (BS, X, Y, FEATURES)
        if self.sensor is not None:
            shape = None
            is_sequence = getattr(self.sensor, "is_sequence_module", False)
            if is_sequence:
                if not is_2d:
                    shape = x.shape[:-1]
                    x = x.reshape(-1, x.shape[-1])
                else:  # (BS, FEATURES, X, Y)
                    shape = x.shape[:-1]
                    if not self.projection.flatten:
                        x = x.transpose(1, 2).transpose(2, 3)  # (BS, X, Y, FEATURES)
                        shape = x.shape[:-1]
                        x = x.reshape(-1, x.shape[-1])  # (BS*X*Y, FEATURES)

            x = self.sensor(x)
            if is_sequence:
                x = x.reshape(*shape, -1)
                if not self.projection.flatten:
                    x = x.transpose(2, 3).transpose(1, 2)  # (BS, FEATURES, X, Y)
        return x

    def reset(self):
        if self.sensor:
            self.sensor.reset()

    def __str__(self):
        projection = f" -> {self.projection.__class__.__name__}_{self.projection_size}" if self.projection else ""
        embed = f" -> {self.sensor.__class__.__name__}_{self.embed_size}" if self.sensor else ""
        return f"{self.__class__.__name__}: input_{self.input_size}{projection}{embed}"

    @property
    def is_sequence_module(self):
        return self.sensor and self.sensor.is_sequence_module

    @property
    def states(self):
        if not self.is_sequence_module:
            return None

        return self.sensor.states

    @states.setter
    def states(self, value):
        assert self.is_sequence_module
        self.sensor.states = value

    @property
    def num_layers(self):
        if self.is_sequence_module:
            return self.sensor.num_layers
        return 0

    @property
    def hidden_size(self) -> Union[int, tuple]:
        if self.is_sequence_module:
            return self.sensor.hidden_size
        return 0


class RegulatorEmbedding(Embedding):
    """ Embedding Module for a single Regulating Parameter. """

    REPR_FIELDS = ("scale", *Embedding.REPR_FIELDS)

    def __init__(self, scale=1., **kwargs):
        """ Constructs an Embedding Instance """
        Module.__init__(self)

        self.p = Parameter(randn(1))
        self.scale = scale
        Embedding.__init__(self, **kwargs)

    def get_regulation_value(self, apply_sigmoid: bool = True):
        """ Returns the regulation value of the module.
        :param apply_sigmoid: (bool) Whether to apply a sigmoid function to the regulation value, or `abs` instead.
        """
        if apply_sigmoid:
            return (tanh(self.p).item() + 1.) * 0.5 * self.scale

        # take absolute value of regulation value
        return torch_abs(self.p).item()

    def set_regulation_value(self, value, apply_sigmoid: bool = True):
        """ Sets the regulation value of the module. """
        value = torch.tensor(value)
        if apply_sigmoid:
            value = torch.arctanh(2. * value / self.scale - 1.)

        self.p.data = value

    def _build(self):
        pass

    def to(self, device):
        Embedding.to(self, device)
        self.p = Parameter(self.p.to(device))
        return self


class PositionalEmbedding(Embedding):
    """ Embedding Module of positional feature input into an embedding of arbitrary length. """

    REPR_FIELDS = ("embedding_size", "total_length", "normalize", "batch_expand",
                   *Embedding.REPR_FIELDS)

    def __init__(self, embedding_size: int, total_length=10000, normalize=False, batch_expand=2, **kwargs):
        """ Constructs a PositionalEmbedding Instance

        :param embedding_size: (int) The size of the embedding.
        :param total_length: (int) The total length of the sequence.
        :param normalize: (bool) Whether to normalize the positions to [0, 1]. This might be useful for
                           interpolation between different positional embeddings.
        :param batch_expand: (int) The number of dimensions of the input tensor, including the batch dimension.
                             If the input tensor has fewer dimensions, it is expanded to `batch_expand` dimensions
                             by adding a batch dimension at the beginning. Dimension 1 then is the sequence length,
                             which is used to compute the positional embedding.
        """
        Module.__init__(self)
        self.embedding_size = embedding_size
        self.total_length = total_length
        self.normalize = normalize
        self.batch_expand = batch_expand
        Embedding.__init__(self, **kwargs)

    def _build(self):
        pass

    def forward(self, x, *args: Tensor) -> Tensor:
        """ Transforms an input, `x`, into a positional embedding, following the formula

            - Embedding[i, 2k] = sin(position / (self.total_length^(2k / self.embedding_size)))
            - Embedding[i, 2k+1] = cos(position / (self.total_length^(2k / self.embedding_size)))

        :param x: (torch.Tensor) input tensor to the neural network of dim (`batch_size`, `sequence_length`, *`input_sizes`).
        :returns: The (`batch_size`, `sequence_length`, `embedding_size`) positional embedding wrt the sequence_length.
        """
        if len(x.shape) < self.batch_expand:
            x = x[None, ...]

        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        embedding_size = self.embedding_size
        n = self.total_length

        positions = torch.arange(sequence_length, device=self.device).float().unsqueeze(1)
        if self.normalize:
            positions /= positions.max()

        embeddings = torch.zeros(sequence_length, embedding_size)

        denominators = torch.pow(n, 2 * torch.arange(0, embedding_size // 2) / embedding_size)  # 10000^(2i/d_model), i is the index of embedding
        embeddings[..., 0::2] = torch.sin(positions / denominators)  # sin(pos/10000^(2i/d_model))
        embeddings[..., 1::2] = torch.cos(positions / denominators)  # cos(pos/10000^(2i/d_model))

        return embeddings[None, ...].repeat(batch_size, 1, 1)


class CPPNStateEmbedding(StateEmbedding):
    """ A module providing a tensor comprising state parameters
        for a number of different possible states through a
        Compositional Pattern-Producing Network (CPPN) for, e.g.,
        the in the initial seed of a developmental phase in an NCA. """
    REPR_FIELDS = ('cppn', 'positional_embedding', 'cat_xp', *StateEmbedding.REPR_FIELDS,)

    def __init__(self,
                 cppn: Union[str, dict, Patchwork],
                 state_size: list,
                 positional_embedding: Optional[Union[str, dict, PositionalEmbedding]] = None,
                 num_states: int = 1,
                 randomize: bool = True,
                 cat_xp: bool = False,
                 **patchwork_kwargs
                 ):
        """

        :param cppn:
        :param state_size:
        :param positional_embedding:
        :param num_states:
        :param randomize:
        :param cat_xp: Whether the CPPN is parameter aware, i.e., a concatenation of the parameters and the
                       corresponding positional encoding is passed to the CPPN, defaults to False (only the
                       positional encoding is passed to the CPPN).
        :param patchwork_kwargs:
        """

        self.cppn = cppn
        self.positional_embedding = positional_embedding
        self.cat_xp = cat_xp

        # torch Parameters
        StateEmbedding.__init__(self, state_size=state_size, num_states=num_states, randomize=randomize,
                                **patchwork_kwargs)

    def _build(self):
        self.cppn = Patchwork.make(self.cppn, is_nested=True)
        if self.positional_embedding is not None:
            self.positional_embedding = PositionalEmbedding.make(self.positional_embedding)

    def to(self, device) -> 'CPPNStateEmbedding':
        Embedding.to(self, device)
        self.cppn = self.cppn.to(device)
        if self.state is not None:
            self.state = self.state.to(device)

        if self.positional_embedding is not None:
            self.positional_embedding = self.positional_embedding.to(device)

        return self

    def forward(self, x: Tensor, *args) -> Tensor:
        """ Returns states according to state-indices/coords specified in `x`, and transformed by the
            CPPN into states (if `to_states` is specified)

        :param x: Coordinate tensor addressing cells in a tissue for which the `CPPN` evaluates the
                  corresponding cell states.
        :return: Tensor of states addressed by `x`.
        """
        if self.positional_embedding is not None:
            p = self.positional_embedding(x)[0]  # assome same positional embedding for all batch elements
            if self.cat_xp:
                if len(x.shape) == 1:
                    x = x[..., None]             # add feature dimension to 1D vector

                p = torch.cat((x, p), dim=-1)

            self.state = self.cppn(p)

        else:
            d = torch.tensor(self.num_states, device=self.device)[None, :]
            self.state = self.cppn(x / d)  # coords to relative coords

        # self.state = torch.argmax(self.embedding, dim=-1)
        return self.state

    @property
    def input_size(self):
        return self.cppn.input_size

    @property
    def embed_size(self) -> Optional[int]:
        return self.cppn.output_size

    @property
    def output_size(self) -> Optional[int]:
        return self.embed_size


if __name__ == '__main__':
    p = PositionalEmbedding(4, total_length=1, normalize=True)
    x = torch.randn(2, 100, 1)
    print(p(x)[0])

    import matplotlib.pyplot as plt
    plt.imshow(p(x)[0].detach().numpy())
    plt.show()
