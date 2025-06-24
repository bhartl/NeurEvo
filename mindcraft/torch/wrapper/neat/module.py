import torch
from torch.nn import Module, Parameter
from . import activation
from . import aggregation
from mindcraft.torch.wrapper.graph import Wiring
from networkx import DiGraph
import numpy as np


class NEATModule(Module):
    """ A PyTorch Wrapper for the `mindcraft.torch.layers.Wiring` class used for `NEATMethod` adaptation

    (c) B. Hartl 2021
    """
    def __init__(self,
                 wiring: (DiGraph, Wiring),
                 input_size=None,
                 output_size=None,
                 stateful=True,
                 **kwargs
                 ):
        Module.__init__(self)

        # hidden variables
        self._wiring = None
        self._input_size = input_size
        self._output_size = output_size

        # setup everything
        self.kwargs = kwargs
        self.wiring = wiring

        # helpers
        self.stateful = stateful
        self.states = None

    @property
    def wiring(self):
        return self._wiring

    @wiring.setter
    def wiring(self, value):
        self._wiring = Wiring.make(value, input_size=self.input_size, output_size=self.output_size, **self.kwargs)
        self.init_neurons()

    def init_neurons(self):
        neuron_activations = {}
        neuron_aggregations = {}

        for node_idx in self._wiring.digraph.nodes:
            # get node by index
            node = self._wiring.digraph.nodes[node_idx]

            try:
                neat_activation = getattr(activation, node['activation'])
                if isinstance(neat_activation, type):  # check for "stateful" activations, which need to be created
                    neat_activation = neat_activation(**node)

                neuron_activations[node_idx] = neat_activation

            except KeyError:  # input nodes may not have activation functions
                pass

            try:
                neat_aggregation = getattr(aggregation, node['aggregation'])
                if isinstance(neat_aggregation, type):  # check for "stateful" aggregations, which need to be created
                    neat_aggregation = neat_aggregation(**node)

                neuron_aggregations[node_idx] = neat_aggregation

            except KeyError:
                # input nodes may not have activation functions
                pass

        self._wiring.activations = neuron_activations
        self._wiring.aggregations = neuron_aggregations

    @property
    def input_size(self):
        if self._wiring is None:
            return self._input_size
        return self._wiring.input['n']

    @property
    def output_size(self):
        if self._wiring is None:
            return self._output_size
        return self._wiring.output['n']

    def hidden_size(self):
        if self._wiring is None:
            return 0
        return self._wiring.hidden['n']

    def forward(self, x, states=None):
        if states is None:
            states = self.states

        y, states = self.wiring.forward(x, states, zeros=torch.zeros, stack=torch.stack, axis_kw='dim')

        if self.stateful:
            self.states = states

        return y, states

    def reset(self):
        self.states = None


class WiredRNN(NEATModule):
    """ A PyTorch Wrapper for a recurrent `mindcraft.torch.layers.Wiring` used for `NEATMethod` adaptation

    (c) B. Hartl 2021
    """
    def __init__(self,
                 wiring: (Wiring, DiGraph),
                 input_size: (int, None) = None,
                 output_size: (int, None) = None,
                 stateful=True,
                 **kwargs
                 ):

        NEATModule.__init__(self, wiring=wiring, stateful=stateful, input_size=input_size, output_size=output_size,
                            **kwargs)
        self._build()

    def _build(self):
        dict_repr = self.wiring.to_dict()

        def add_param(attrs, key, name):
            try:
                value = attrs[key]
                assert hasattr(value, '__iter__')
                value = torch.Tensor(value)
            except AssertionError:
                value = torch.Tensor(np.asarray([attrs[key]]))
            except KeyError:
                return

            param = Parameter(value)
            setattr(self, f'{name}_{key}', param)
            attrs[key] = param
            return param

        # transform node attributes to torch tensor and add as network parameters
        nodes = dict_repr['nodes']
        for key in nodes.keys():
            add_param(attrs=nodes[key], key='bias', name=f'node_{key}')
            add_param(attrs=nodes[key], key='response', name=f'node_{key}')  # todo: remove if not response mutation

        # transform edges attributes as torch parameters
        edges = dict_repr['edges']
        for u, edges_u in edges.items():
            for v, attrs_uv in edges_u.items():
                if attrs_uv == {}:
                    continue

                add_param(attrs=attrs_uv, key='weight', name=f'edge_{u}_{v}')

        self.wiring = Wiring.make(dict_repr)

    def summary(self):
        _str = "|" + ('='*68) + "|\n"
        _str += "|" + f'{"param-name":>24s} |{"tensor-shape":>24s} |{"#trainable":>16s}' + "|\n"
        _str += "|" + ('-'*24) + " |" + ('-'*24) + " |" + ('-'*16) + "|\n"

        input_str = f'input[-{self.input_size}:-1]'
        _str += "|" + f'{input_str:>24s} |{str(((None, (None,), self.input_size))):>24s} |{"0":>16s}' + "|\n"
        _str += "|" + ('-'*24) + " |" + ('-'*24) + " |" + ('-'*16) + "|\n"

        n_trainable = 0
        n_nodes = 0
        for n, w in self.named_parameters():
            shape = str((None, (None,), *w.shape))
            trainables = np.prod(w.shape)
            n_trainable += trainables
            n_nodes += 1
            _str += "|" + f'{n:>24s} |{shape:>24s} |{str(trainables):>16s}' + "|\n"
        _str += "|" + ('-'*24) + " |" + ('-'*24) + " |" + ('-'*16) + "|\n"

        _str += "|" + f"{f'total: {n_nodes}':>24}" + " |" + f"{str(((None, (None,), self.output_size))):>24s}" + " |" \
                + f'{str(n_trainable):>16s}' + "|\n"
        _str += "|" + '='*68 + "|\n"
        return _str

    def __str__(self):
        name, n_in, n_out = self.__class__.__name__, self.input_size, self.output_size
        return name + f"(dim_in={n_in}, dim_out={n_out})"
