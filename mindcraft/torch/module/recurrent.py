from torch import Tensor, zeros
from torch.nn import Module, Identity, Linear
from mindcraft.torch.module import Patchwork
from mindcraft.torch.util import get_torch_layer, layer_type_repr
from typing import Union, Optional


class Recurrent(Patchwork):
    """ RNN-based PyTorch Module implementation of the `mindcraft` Framework. """

    REPR_FIELDS = (
            "input_size",
            "hidden_size",
            "num_layers",
            "output_size",
            "stateful",
            "return_sequence",
            "layer_type",
            "layer_kwargs",
            "add_batch_dim",
            *Patchwork.REPR_FIELDS,
        )

    def __init__(self,
                 input_size: int,
                 hidden_size: int = None,
                 num_layers: int = 1,
                 output_size: int = None,
                 stateful=True,
                 return_sequence=False,
                 layer_type: Union[str, Module] = 'LSTM',
                 layer_kwargs: Optional[dict] = None,
                 add_batch_dim: bool = False,
                 **patchwork_kwargs
                 ):
        """ Initialize a Recurrent PyTorch Module.

        :param input_size: The Feature Dimension of the Input.
        :param hidden_size: The Hidden Dimension of the RNN.
        :param num_layers: The Number of Layers of the RNN.
        :param output_size: The Feature Dimension of the Output.
        :param stateful: A Boolean indicating whether the RNN should be stateful.
        :param return_sequence: A Boolean indicating whether the RNN should return a sequence or only the last output.
        :param layer_type: The Type of the RNN Layer as string, naming the PyTorch Module.
        :param layer_kwargs: The Keyword Arguments for the corresponding RNN Layer.
        :param add_batch_dim: A Boolean indicating whether the RNN should add a Batch Dimension to the Input if
                              only a two-dimensional Tensor is passed, otherwise it will add a single Sequence
                              Dimension.
        :param patchwork_kwargs: Keyword Arguments forwarded to the Patchwork.
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.stateful = stateful
        self.return_sequence = return_sequence
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        self.add_batch_dim = add_batch_dim

        # torch module
        self.rnn = None
        self.head = None

        patchwork_kwargs["omit_default"] = patchwork_kwargs.get("omit_default", True)
        Patchwork.__init__(self, **patchwork_kwargs)

        # rnn - helpers
        self._states = None

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        if 'layer_type' in dict_repr:
            dict_repr['layer_type'] = layer_type_repr(self.layer_type)

        return dict_repr

    def _build(self):
        # load layer type
        layer_type: type = get_torch_layer(self.layer_type)
        layer_kwargs = self.layer_kwargs or {}
        layer_kwargs = {k: v for k, v in layer_kwargs.items()}
        layer_kwargs['batch_first'] = layer_kwargs.get('batch_first', True)

        # determine layout
        if self.num_layers > 1:
            if self.hidden_size is None:
                self.hidden_size = self.input_size

            if self.output_size is None:
                self.output_size = self.hidden_size

        elif self.num_layers == 1:
            if self.output_size is None:
                self.output_size = self.input_size

            if self.hidden_size is None:
                self.hidden_size = self.output_size

        else:
            raise ValueError(f"`num_layers` must be `>=1`, got {self.num_layers}.")

        self.rnn = layer_type(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              **layer_kwargs)

        self.head = Identity()
        if self.num_layers == 1:
            if self.layer_type == "StackRNN" and self.output_size != self.input_size:
                self.head = Linear(self.input_size, self.output_size)

            elif self.output_size != self.hidden_size:
                self.head = Linear(self.hidden_size, self.output_size)

        elif not (self.input_size == self.hidden_size == self.output_size):
            if self.layer_type == "StackRNN":
                self.head = Linear(self.input_size, self.output_size)
            else:
                self.head = Linear(self.hidden_size, self.output_size)

    def forward(self, x, *args, states=None) -> Tensor:
        # merge flattened features, but keep batch-size and seq-len,
        x = Patchwork.forward(self, x, *args)
        is_sequence = len(x.shape) > 2
        if not is_sequence:  # add sequence length
            # (batch_size, seq_len == 1, features) if batch_first
            # (seq_len == 1, batch_size, features) otherwise
            if self.add_batch_dim:
                batch_dim = 0 if self.rnn.batch_first else 1
                x = x.unsqueeze(batch_dim)
            else:
                seq_dim = 1 if self.rnn.batch_first else 0
                x = x.unsqueeze(seq_dim)

        # evaluate lstm (with potential former hidden states)
        x, states = self.rnn(x, self.rnn_states if self.stateful else states)
        if self.stateful:
            self.states = states

        if not self.return_sequence or (not is_sequence and not self.return_sequence):
            x = x[:, -1] if self.rnn.batch_first else x[-1]

        x = self.head(x)
        return x

    @property
    def hidden_state(self):
        if self.states is None:
            return zeros(self.num_layers, self.hidden_size)

        return self.states[0].squeeze(1)  # tuple (hidden_state, cell_state) for LSTM

    @property
    def rnn_states(self):
        return self._states

    @property
    def states(self):
        if self._states is None:
            return None

        if not isinstance(self._states, tuple):
            return self._states,  # return as tuple

        return self._states

    @states.setter
    def states(self, value):
        if isinstance(value, list):
            value = tuple(value)

        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]

        self._states = value

    def reset(self):
        self._states = None
        if hasattr(self.rnn, 'reset'):
            self.rnn.reset()

    def to(self, device) -> 'Recurrent':
        Patchwork.to(self, device)
        self.rnn = self.rnn.to(device)
        self.head = self.head.to(device)

        if hasattr(self, 'states') and self.states is not None:
            states = [(state.to(device) for state in self.states)]
            # if len(states) == 1:  # RNN, GRU
            #     self.states = states[0]
            # else:  # LSTM
            self.states = tuple(states)

        return self

    @property
    def is_sequence_module(self):
        return True

    def __str__(self):
        return f"{self.__class__.__name__}[{self.layer_type}]({self.input_size},{self.hidden_size},{self.output_size})"
