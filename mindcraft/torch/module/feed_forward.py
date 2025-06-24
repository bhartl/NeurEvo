from torch.nn import Sequential, Linear, Dropout, BatchNorm1d
from torch import Tensor
from mindcraft.torch.module import Patchwork
from typing import Optional, Union
from mindcraft.torch.activation import get_activation_function
from numpy import isscalar


class FeedForward(Patchwork):
    """ (Multilayer) configurable Sequential PyTorch Module of the `mindcraft` framework

    (c) B. Hartl 2021
    """

    REPR_FIELDS = (
            "input_size",
            "hidden_size",
            "output_size",
            "bias",
            "batch_norm",
            "activation",
            "dropout",
            *Patchwork.REPR_FIELDS,
        )
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: Optional[Union[int, list, tuple]] = None,
                 activation: Optional[Union[list, tuple, str]] = None,
                 bias: Optional[Union[list, tuple, bool]] = True,
                 batch_norm: Optional[Union[list, tuple, bool]] = None,
                 dropout: Optional[Union[list, tuple, float]] = None,
                 **patchwork_kwargs,
                 ):
        """ Constructs a `FeedForward` instance

        :param input_size: Integer specifying the number of input dimensions to the network.
        :param output_size: Integer specifying the number of output dimensions of the network.
        :param hidden_size: Integer or tuple/list of integers specifying the successive dimensions of hidden layers
                            of the network.
        :param activation: Optional representation or list of representations of PyTorch activation
                           functions (e.g. "ReLU", ...) that are element wise retrieved via
                           `mindcraft.torch.activation.get_activation_function(...)`. If a single activation name
                           is provided, all layers are activated with the same function. If several are provided,
                           each layer will have its own activation. Note that `None` is a valid option.
                           Defaults to None.
        :param bias: Optional boolean flag or list of boolean flag to either globally or layer-wise enable/disable
                     the use of biases. Defaults to True.
        :param batch_norm: Optional boolean flag or list of boolean flag to either globally or layer-wise enable/disable
                           the use of BatchNorm layers after the network application but prior to activation and dropout.
                           Note that `None` is a valid option, boolean values will be used as `affine` arguments in the
                           BatchNorm initializations, respectively. Defaults to None.
        :param dropout: Optional float or list of float values to either globally or layer-wise define the dropout
                        fraction after the network activation, defaults to None.
        :param patchwork_kwargs: Keyword-Args to be forwarded to the `Patchwork` base-class constructor.
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation        
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.nn = None

        patchwork_kwargs['omit_default'] = patchwork_kwargs.get('omit_default', True)
        Patchwork.__init__(self, **patchwork_kwargs)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        
        hidden_size = dict_repr.get('hidden_size', None)
        if hidden_size is not None and not isscalar(hidden_size):
            dict_repr['hidden_size'] = list(hidden_size)

        for str_attr in ["activation"]:
            attr = dict_repr.get(str_attr, None)
            if attr is not None and not isinstance(attr, str):
                dict_repr[str_attr] = list(attr)

        for bool_attr in ["batch_norm", "bias"]:
            attr = dict_repr.get(bool_attr, None)
            if attr is not None and not isinstance(attr, bool):
                dict_repr[bool_attr] = list(attr)

        for float_attr in ["dropout"]:
            attr = dict_repr.get(float_attr, None)
            if attr is not None and not isinstance(attr, float):
                dict_repr[float_attr] = list(attr)

        return dict_repr

    def _build(self, ):
        # determine number of layers in network, based on hidden_size definition
        layer_size = []
        self.num_hidden = 0
        if self.hidden_size:
            self.num_hidden = 1 if isscalar(self.hidden_size) else len(self.hidden_size)
            hidden_size = [self.hidden_size] if not hasattr(self.hidden_size, '__iter__') else self.hidden_size  # list
            layer_size = [h for h in hidden_size]

        # add final layer
        layer_size.append(self.output_size)
        num_layers = len(layer_size)

        # determine bias for each layer
        bias = self.bias
        if not isinstance(bias, bool):
            assert num_layers == len(bias), f"Got {len(bias)} bias values for {num_layers} layers."

        else:
            bias = [self.bias] * num_layers

        # determine batch_norm for each layer
        batch_norm = self.batch_norm
        if batch_norm is not None and not isinstance(batch_norm, bool):
            assert num_layers == len(batch_norm), f"Got {len(batch_norm)} batch_norm values for {num_layers} layers."
            batch_norm = [BatchNorm1d(layer_size[i], affine=a) if a is not None else None
                          for i, a in enumerate(self.batch_norm)]
        else:
            batch_norm = [BatchNorm1d(layer_size[i], affine=self.batch_norm) if self.batch_norm is not None else None
                          for i in range(len(layer_size))]

        # determine activation for each layer
        layer_specific_activation = self.activation is not None and not isinstance(self.activation, str)
        if layer_specific_activation:
            num_activations = len(self.activation)
            assert num_layers == num_activations, f"Got {num_activations} activations for {num_layers} layers."
            activation = [get_activation_function(a) for a in self.activation]
            
        else:
            activation = [get_activation_function(self.activation) for _ in range(len(layer_size))]

        # determine dropout for each layer
        dropout = self.dropout
        if dropout is not None and not isinstance(dropout, float):
            assert num_layers == len(dropout), f"Got {len(dropout)} dropout values for {num_layers} layers."
            dropout = [(Dropout(d) if d is not None else None) for d in self.dropout]
        else:
            dropout = [Dropout(self.dropout) if self.dropout is not None else None] * num_layers

        # compose sequential model
        layers = []
        layer_size = [self.input_size] + layer_size  # add first layer for convenience:
        for num_in, num_out, b, bn, foo, d in zip(layer_size[:-1],
                                                  layer_size[1:],
                                                  bias,
                                                  batch_norm,
                                                  activation,
                                                  dropout):
            try:
                layers.append(Linear(in_features=num_in, out_features=num_out, bias=b))
            except TypeError as e:
                print(e)
                raise
            if bn is not None:  # only add if batch_norm is defined
                layers.append(bn)

            if foo is not None:  # only add if activation function is not None
                layers.append(foo)

            if d is not None:  # only add if dropout is not None
                layers.append(d)

        self.nn = Sequential(*layers)

    def forward(self, x, *args) -> Tensor:
        # merge flattened features, but keep batch-size and seq-len,
        x = Patchwork.forward(self, x, *args)
        return self.nn(x)
