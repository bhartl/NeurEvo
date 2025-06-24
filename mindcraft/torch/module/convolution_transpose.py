from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.nn import Sequential, Dropout, Flatten
from torch import Tensor, randn
from mindcraft.torch.module import Patchwork, FeedForward
from typing import Optional, Union
from mindcraft.torch.activation import get_activation_function
from functools import partial
from mindcraft.torch.util import get_conv_transpose2d_output_size


class ConvT(Patchwork):
    """ (Multilayer) configurable Sequential Convolutional PyTorch Module of the `mindcraft` framework

    (c) B. Hartl 2021
    """
    REPR_FIELDS = (
            "input_size",
            "input_dim",
            "expand_size",
            "filters",
            "kernel_size",
            "strides",
            "padding",
            "dilation",
            "padding_mode",
            "output_padding",
            "batch_norm",
            "activation",
            "dropout",
            "flatten",
            *Patchwork.REPR_FIELDS,
        )
    
    def __init__(self,
                 input_size: int,
                 input_dim: int,
                 filters: Union[list, tuple],
                 kernel_size: Union[list, tuple],
                 expand_size: int = None,
                 strides: Optional[Union[list, tuple]] = None,
                 padding: Union[int, tuple] = 0,
                 output_padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 padding_mode: str = 'zeros',
                 activation: Optional[Union[list, tuple, str]] = None,
                 batch_norm: Optional[Union[list, tuple, bool]] = None,
                 dropout: Optional[Union[list, tuple, float]] = None,
                 flatten: Optional[Union[Patchwork, bool, dict]] = None,
                 **patchwork_kwargs,
                 ):
        """ Constructs a Convolution instance

        see https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html for more information.

        :param input_size: Number of input channels.
        :param input_dim:  Integer specifying the number of input dimensions, i.e., 1D, 2D and 3D convolutions.
        :param expand_size: Optional integer specifying the dimension of an expansion-projection from
                            the input space, defaults to None.
        :param filters: List or tuple of integers, specifying the number of filters per conv layer
        :param kernels_size: List or tuple of kernel sizes for each conv layer
        :param strides: Optional list or tuple of stride values for each conv layer
        :param activation: String-Name or list of names of activation functions to be used in each conv layer
                           (defaults to 'relu').
                           If a single name is specified, the activation is used for each layer.
                           If None is provided, no activation is used.
        :param padding: Padding of all conv layers (defaults to 0, global property).
        :param dilation: Controls the spacing between the kernel points; also known as the à trous algorithm,
                         defaults to 1. (see https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
        :param output_padding: Additional size added to one side of the output shape (defaults to 0).
        :param padding_mode: Padding mode (defaults to 'zeros', see torch doc).
        :param flatten: Optional (i) boolean or (ii) PatchWork representation to (i) flatten the output of the last CNN
                        layer and (ii) additionally apply another PatchWork transformation (e.g., with a dense
                        FeedForward layer), defaults to None.
        :param kwargs: Keyword Arguments forwarded to the super constructor (see Encoder).
        """
        self.input_size = input_size
        assert input_dim in (1, 2, 3)
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.flatten = flatten
        self.expand_size = expand_size
        self.cnn = None

        patchwork_kwargs['omit_default'] = patchwork_kwargs.get('omit_default', True)
        to_list = list(patchwork_kwargs.get('to_list', []))
        [to_list.append(li)
         for li in ["filters", "kernel_size", "strides", ]
         if li not in to_list]
        patchwork_kwargs['to_list'] = to_list
        Patchwork.__init__(self, **patchwork_kwargs)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)

        for str_attr in ["activation"]:
            attr = dict_repr.get(str_attr, None)
            if attr is not None and not isinstance(attr, str):
                dict_repr[str_attr] = list(attr) if hasattr(attr, "__iter__") else attr

        for bool_attr in ["batch_norm"]:
            attr = dict_repr.get(bool_attr, None)
            if attr is not None and not isinstance(attr, bool):
                dict_repr[bool_attr] = list(attr) if hasattr(attr, "__iter__") else attr

        for float_attr in ["dropout"]:
            attr = dict_repr.get(float_attr, None)
            if attr is not None and not isinstance(attr, float):
                dict_repr[float_attr] = list(attr) if hasattr(attr, "__iter__") else attr

        return dict_repr

    def _build(self, ):
        # determine number of layers in network, based on hidden_size definition
        self.num_layers = len(self.filters) + (self.expand_size is not None)
        self.num_hidden = self.num_layers - 1

        # determine batch_norm for each layer
        batch_norm = self.batch_norm

        def get_batch_norm(layer_index, layer_type):
            if layer_type is None:
                return None

            if self.input_dim == 1:
                return BatchNorm1d(self.filters[layer_index], affine=layer_type)

            elif self.input_dim == 2:
                return BatchNorm2d(self.filters[layer_index], affine=layer_type)

            return BatchNorm3d(self.filters[layer_index], affine=layer_type)

        if batch_norm is not None and not isinstance(batch_norm, bool):
            assert self.num_layers == len(batch_norm), f"Got {len(batch_norm)} batch_norm values " \
                                                       f"for {self.num_layers} layers."
            batch_norm = [get_batch_norm(i, a) for i, a in enumerate(self.batch_norm)]
        else:
            batch_norm = [get_batch_norm(i, self.batch_norm) for i in range(self.num_layers)]

        # determine activation for each layer
        layer_specific_activation = self.activation is not None and not isinstance(self.activation, str)
        if layer_specific_activation:
            num_activations = len(self.activation)
            assert self.num_layers == num_activations, f"Got {num_activations} activations for {self.num_layers} layers."
            activation = [get_activation_function(a) for a in self.activation]
        else:
            activation = [get_activation_function(self.activation) for _ in range(self.num_layers)]

        # determine dropout for each layer
        dropout = self.dropout
        if dropout is not None and not isinstance(dropout, (float, bool)):
            assert self.num_layers == len(dropout), f"Got {len(dropout)} dropout values for {self.num_layers} layers."
            dropout = [(Dropout(d) if d is not None else None) for d in self.dropout]
        else:
            dropout = [Dropout(self.dropout) if self.dropout is not None else None] * self.num_layers

        # compose sequential model
        if self.input_dim == 1:
            ConvT_nd = partial(ConvTranspose1d, padding_mode=self.padding_mode, dilation=self.dilation)
        elif self.input_dim == 2:
            ConvT_nd = partial(ConvTranspose2d, padding_mode=self.padding_mode, dilation=self.dilation)
        elif self.input_dim == 3:
            ConvT_nd = partial(ConvTranspose3d, padding_mode=self.padding_mode, dilation=self.dilation)
        else:
            raise ValueError(self.input_dim)

        layers = []

        input_size = self.input_size
        if self.expand_size:
            layers.append(FeedForward(input_size=input_size, output_size=self.expand_size,
                                      batch_norm=batch_norm[0], activation=activation[0], dropout=[0]))
            batch_norm, activation, dropout = batch_norm[1:], activation[1:], dropout[1:]
            input_size = self.expand_size

        in_channels = [input_size] + list(self.filters[:-1])
        strides = self.strides or [1] * self.num_layers
        padding = self.padding if hasattr(self.padding, "__iter__") else [self.padding] * self.num_layers
        output_padding = self.output_padding if hasattr(self.output_padding, "__iter__") else [self.output_padding] * self.num_layers
        for in_c, f, k, s, p, op, bn, foo, d in zip(in_channels,
                                                    self.filters,
                                                    self.kernel_size,
                                                    strides,
                                                    padding,
                                                    output_padding,
                                                    batch_norm,
                                                    activation,
                                                    dropout):
            layers.append(ConvT_nd(in_channels=in_c, out_channels=f, kernel_size=k, stride=s, padding=p, output_padding=op))
            if bn is not None:  # only add if batch_norm is defined
                layers.append(bn)

            if foo is not None:  # only add if activation function is not None
                layers.append(foo)

            if d is not None:  # only add if dropout is not None
                layers.append(d)

        if self.flatten is not None:
            if isinstance(self.flatten, bool):
                if self.flatten:
                    layers.append(Flatten())
            else:  # assume Patchwork
                layers.append(Flatten())
                layers.append(Patchwork.make(self.flatten))

        self.cnn_t = Sequential(*layers)

    def forward(self, x, *args) -> Tensor:
        if len(x.shape) <= (self.input_dim + 1):
            for _ in range(self.input_dim):
                if self.input_dim == 1:
                    x = x.unsqueeze(1)  # add channel-dim
                else:
                    x = x.unsqueeze(-1)  # add channel-dim

        return self.cnn_t(x, *args)

    @classmethod
    def get_output_size(self, input_size, kernel_size, stride, padding=None, output_padding=None):
        size = input_size
        cnn_output_size = []
        if padding is None:
            padding = [0] * len(kernel_size)
        if output_padding is None:
            output_padding = [0] * len(kernel_size)

        for k, s, p, op in zip(kernel_size, stride, padding, output_padding):
            size = int(get_conv_transpose2d_output_size(input_size=size, kernel_size=k, stride=s, padding=p,
                                                        out_padding=op, dilation=1))
            cnn_output_size.append(size)
        return tuple(cnn_output_size)

    def print_tensor_dims(self, input_shape, batch_size=10):
        x = randn(batch_size, *input_shape, device=self.device)
        print("\n{0:<22s}".format("encoder input:"), x.shape)
        for layer in self.cnn_t:
            x = layer(x)
            print("- {0:<20s}".format(str(layer.__class__.__name__)), x.shape)
