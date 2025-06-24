import os.path
from neat import genome
from torch import Tensor, zeros
from torch.nn import Module, Identity, Linear
from mindcraft.io import Repr
from mindcraft.torch.module import Patchwork
from mindcraft.torch.util import get_torch_layer, layer_type_repr
from typing import Union, Optional
from mindcraft.torch.wrapper.neat.module import WiredRNN, Wiring
from mindcraft.train.neat_util import ConfigWrapper


class NEATWiring(Patchwork):
    """ RNN-based PyTorch Module implementation of the `mindcraft` Framework. """

    REPR_FIELDS = (
            "input_size",
            "output_size",
            "return_sequence",
            "wiring",
            # "wiring_args",
            "prune_unused",
            *Patchwork.REPR_FIELDS,
        )

    def __init__(self,
                 input_size: int,
                 output_size: int = None,
                 return_sequence: bool = False,
                 wiring: Union[str, Wiring] = None,
                 wiring_args: tuple = (),
                 prune_unused: bool = True,
                 **patchwork_kwargs
                 ):

        self._input_size = input_size
        self._output_size = output_size
        self.return_sequence = return_sequence
        self.prune_unused = prune_unused

        # torch module
        self.wiring = wiring
        self.wiring_args = list(wiring_args)
        self.wired_rnn = None
        self._states = None

        patchwork_kwargs["omit_default"] = patchwork_kwargs.get("omit_default", True)
        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self,):
        io_kwargs = dict(input_size=self.input_size, output_size=self.output_size)
        if self.wiring is not None:
            args = []
            for w_arg in self.wiring_args:
                try:
                    assert not isinstance(w_arg, Repr)
                    w_arg = Repr.make(w_arg)
                except:
                    pass
                args.append(w_arg)

            self.wiring = Wiring.make(self.wiring, *args, prune_unused=self.prune_unused, **io_kwargs)
            self.wired_rnn = WiredRNN(wiring=self.wiring, stateful=True, **io_kwargs)
            for parameter in self.parameters():
                parameter.requires_grad = getattr(self, 'retain_grad', False)

        return self.wired_rnn

    @property
    def input_size(self):
        if self.wired_rnn is not None:
            self._input_size = self.wired_rnn.input_size
            return self.wired_rnn.input_size

        return self._input_size

    @property
    def hidden_size(self):
        if self.wired_rnn is not None:
            return self.wired_rnn.hidden_size

        return 0

    @property
    def output_size(self):
        if self.wired_rnn is not None:
            self._output_size = self.wired_rnn.output_size
            return self.wired_rnn.output_size

        return self._output_size

    def forward(self, *x: Tensor) -> Tensor:
        # merge flattened features, but keep batch-size and seq-len,
        x = Patchwork.forward(self, *x)
        x, self._states = self.wired_rnn(x, states=self.states)

        if not self.return_sequence and len(x.shape) > 3:
            return x[:, -1]

        return x

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
        self.wired_rnn.reset()

    def to(self, device) -> 'NEATWiring':
        Patchwork.to(self, device)
        if self.wired_rnn is not None:
            self.wired_rnn = self.wired_rnn.to(device)

        if hasattr(self, 'states') and self.states is not None:
            states = [(state.to(device) for state in self.states)]
            # if len(states) == 1:  # RNN, GRU
            #     self.states = states[0]
            # else:  # LSTM
            self.states = tuple(states)

        return self

    @property
    def is_sequence_module(self):
        return False  # self.wired_rnn.stateful  # TODO

    def deserialize_parameters(self, serialized: Union[str, tuple, list]):
        if serialized is None:
            self.wiring = None
            self.wired_rnn = None
            return None

        if isinstance(serialized, str) and os.path.exists(serialized) and serialized.endswith('.pkl'):
            with open(serialized, 'rb') as f:
                import pickle
                serialized = pickle.load(f)

        if not isinstance(serialized, tuple) and hasattr(serialized, '__len__'):
            if len(serialized) == 2:
                if isinstance(serialized[0], genome.DefaultGenome):
                    if isinstance(serialized[1], ConfigWrapper):
                        serialized = tuple(serialized)

        if not isinstance(serialized, (genome.DefaultGenome, dict, Wiring, tuple)):
            return Patchwork.deserialize_parameters(self, serialized)

        if isinstance(serialized, (tuple)):
            self.wiring, *self.wiring_args = serialized
        else:
            self.wiring = serialized

        self.wired_rnn = self._build()
        return self.wired_rnn

    def serialize_parameters(self, to_numpy=True):
        if self.wiring is None:
            return None

        return self.wiring.to_dict()

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        if self.is_nested:
            dict_repr.pop('recover_indices', None)
            dict_repr.pop('serialized', None)
            dict_repr.pop('serialize_mask', None)
        else:
            dict_repr['serialized'] = self.serialize_parameters(to_numpy=True)

        if self.wiring is not None:
            dict_repr['wiring'] = self.wiring.to_dict()
        else:
            dict_repr['wiring'] = None

        dict_repr['wiring_args'] = [(wa.to_dict() if hasattr(wa, "to_dict") else wa) for wa in self.wiring_args]
        return dict_repr

    def __str__(self):
        d = self.to_dict()
        try:
            d['hidden'] = len(d['serialized']['nodes']) - self.input_size - self.action_size
        except (TypeError, KeyError):
            pass

        try:
            d['edges'] = len(d['serialized']['edges'])
        except (TypeError, KeyError):
            pass

        d.pop('serialized')

        return f"{self.__class__.__name__}: {'{'}{', '.join(f'{k}:{v}' for k, v in d.items())}{'}'}"
