import tempfile
import networkx as nx
from networkx import Graph, DiGraph
from neat import Config
from typing import Union
from functools import partial
from mindcraft.io.spaces import space_clip
from numpy import sum as np_sum
from numpy import prod as np_prod
from numpy import exp as np_exp
from numpy import stack as np_stack
from numpy import zeros as np_zeros


SERIALIZE_NODE_ATTRS = ['bias', 'aggregation', 'activation', 'response']
SERIALIZE_EDGE_ATTRS = ['enabled', 'weight']


def get_nested_attr(node, edge):
    """ retrieve nested attribute in node along `edge` recursively calling
        `get_nested_attr(node[edge[0]], edge[1:])` until a node is reached where `len(edge) == 0`.

    (c) B. Hartl 2021

    :param node:
    :param edge:
    :return: The final `node` along the `edge`.
    """
    if len(edge) == 0:
        return node
    return get_nested_attr(node[edge[0]], edge[1:])


def split_cyclic(g: nx.DiGraph, input_nodes, output_nodes, recurrent=None):
    """ Try to identify a cylce in a `nx.DiGraph` object, and split it to get a feed forward graph

    (c) B. Hartl 2021

    :param g:
    :param input_nodes:
    :param output_nodes:
    :param recurrent:
    :return:
    """
    try:
        r = nx.find_cycle(g, orientation="original")
        if recurrent is None:
            recurrent = nx.DiGraph()

        while r:
            i = -1
            backward = r[i]  # (in-node, out-node, mode)
            while backward[1] not in output_nodes and len(r) > abs(i):
                i -= 1
                backward = r[i]

            for node in backward[:2]:  # (u, v, type[forward/backward])
                attr = nx.get_node_attributes(g, node)
                recurrent.add_node(node, **attr)

            attr = g.get_edge_data(*backward[:2])
            recurrent.add_edge(*backward[:2], **attr)

            g.remove_edge(*backward[:2])
            r = nx.find_cycle(g, orientation="original")

    except nx.NetworkXNoCycle:
        pass  # no cycles anymore

    return g, recurrent


def required_for_output(inputs, outputs, connections):
    """ Adapted from neat-python's neat.graphs under the BSD 3-Clause Licence (Sept. 2021)

    Collect the nodes whose state is required to compute the final wired_rnn output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the wired_rnn.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.

    (c) B. Hartl 2021
    """

    required = set(outputs)
    s = set(outputs)
    while True:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def digraph_to_neat_dict(g, input_size, output_size):
    """ Transform a `nx.DiGraph` instance into a `neat`-dictionary

    (c) B. Hartl 2021
    """
    from collections import OrderedDict
    genome = dict(nodes=OrderedDict(), connections=OrderedDict())
    nodes = genome['nodes']
    n_input = 0
    for key in g.nodes:
        node = g.nodes[key]
        if key < 0:  # input
            nodes[key] = dict()
            n_input += 1
        else:
            nodes[key] = {attr: node[attr] for attr in SERIALIZE_NODE_ATTRS}
    assert n_input == input_size

    connections = genome['connections']
    for edge in g.edges:
        edge_attr = get_nested_attr(g, edge)
        connections[edge] = dict(**edge_attr, key=edge)

    genome_config = dict(input_keys=sorted([k for k in genome['nodes'].keys() if k < 0]),
                         output_keys=sorted([k for k in genome['nodes'].keys() if 0 <= k < output_size]))
    config = dict(genome_config=genome_config)

    return genome, config


def dict_to_digraph(dict_repr, input_size=None, output_size=None,):
    """ parse dict of form {nodes: {attr: node_attr}, edges: {attr: edge_attr}} to DiGraph object

    - node_attr must be ['bias', 'response', 'aggregation', 'activation'] for non-input nodes (i.e. nodes with key > 0)
    - edge_attr must be ['enabled', 'weight']

    this also works for other frameworks, not only pytorch.

    (c) B. Hartl 2021
    """

    dict_repr = {k: v for k, v in dict_repr.items()}  # copy

    if input_size is None:
            input_size = dict_repr.pop('input_size')

    if output_size is None:
        output_size = dict_repr.pop('output_size')

    g = DiGraph()
    for key, attrs in dict_repr['nodes'].items():
        if key > 0:  # not an input node
            assert all(a in attrs for a in SERIALIZE_NODE_ATTRS)

        g.add_node(key, **attrs)

    edges = []
    for edge_key in ['edges', 'connections']:
        if edge_key in dict_repr:
            edges = dict_repr[edge_key]
            break

    for conn, attrs in edges.items():
        try:  # assume form {(key_i, key_j): attrs}
            key_in, key_out = conn
            assert all(a in attrs for a in SERIALIZE_EDGE_ATTRS)
            g.add_edge(key_in, key_out, **attrs)

        except TypeError:
            # assume form {key_i: {key_j: attrs}, ...}
            key_in = conn
            for key_out, attrs_adj in attrs.items():
                if attrs_adj == {}:
                    continue

                assert all(a in attrs_adj for a in ['enabled', 'weight'])
                g.add_edge(key_in, key_out, **attrs_adj)

    return g, (input_size, output_size)


def neat_to_digraph(genome, config):
    """ parse tensor-transformable wiring information from neat genome

    this also works for other frameworks, not only pytorch.

    (c) B. Hartl 2021
    """

    def get_key_attr(node, attr):
        try:
            attr_generator = getattr(node, attr)
        except AttributeError:
            attr_generator = node[attr]

        if hasattr(attr_generator, '__call__'):
            return attr_generator()
        return attr_generator

    genome_config = get_key_attr(config, 'genome_config')
    input_units, hidden_units, output_units = {}, {}, {}

    # parse keys
    input_units['keys'] = sorted(list(get_key_attr(genome_config, 'input_keys')))
    output_units['keys'] = sorted(list(get_key_attr(genome_config, 'output_keys')))

    genome_nodes = get_key_attr(genome, 'nodes')

    hidden_units['keys'] = sorted([k for k in genome_nodes.keys()
                                   if k not in input_units['keys'] and k not in output_units['keys']])

    # parse responses
    hidden_units['response'] = [get_key_attr(genome_nodes[k], 'response') for k in hidden_units['keys']]
    output_units['response'] = [get_key_attr(genome_nodes[k], 'response') for k in output_units['keys']]

    # parse aggregation
    hidden_units['aggregation'] = [get_key_attr(genome_nodes[k], 'aggregation') for k in hidden_units['keys']]
    output_units['aggregation'] = [get_key_attr(genome_nodes[k], 'aggregation') for k in output_units['keys']]

    # parse activation
    hidden_units['activation'] = [get_key_attr(genome_nodes[k], 'activation') for k in hidden_units['keys']]
    output_units['activation'] = [get_key_attr(genome_nodes[k], 'activation') for k in output_units['keys']]

    # parse biases
    hidden_units['bias'] = [get_key_attr(genome_nodes[k], 'bias') for k in hidden_units['keys']]
    output_units['bias'] = [get_key_attr(genome_nodes[k], 'bias') for k in output_units['keys']]

    g = DiGraph()
    for units in [input_units, hidden_units, output_units]:
        for i, key in enumerate(units['keys']):
            attrs = {k: v[i] for k, v in units.items() if k != 'keys'}
            g.add_node(key, **attrs)

    try:
        genome_connections = get_key_attr(genome, 'connections')
        for conn in genome_connections.values():
            key_in, key_out = get_key_attr(conn, 'key')
            enabled = get_key_attr(conn, 'enabled')
            weight = get_key_attr(conn, 'weight')
            g.add_edge(key_in, key_out, weight=weight, enabled=enabled)

    except KeyError:
        genome_edges = get_key_attr(genome, 'edges')
        for key_in, edge in genome_edges.items():
            for key_out, conn in edge.items():
                enabled = get_key_attr(conn, 'enabled')
                weight = get_key_attr(conn, 'weight')
                g.add_edge(key_in, key_out, weight=weight, enabled=enabled)

    return g, (len(input_units['keys']), len(output_units['keys']))


def attr_to_primitive(value):
    """ Convert numpy array into list or scalar (important for serialization), retains other types.

    (c) B. Hartl 2021
    """
    try:
        try:
            # convert np.float32/64... into native scalar
            return value.item()

        except ValueError:
            # if value is not scalar, try same with np.ndarray
            return value.tolist()

    except AttributeError:
        # default behaviour, if neither of the above
        return value


def digraph_to_dict(g, input_size, output_size):
    """ Parse a digraph Wiring into dict-repr with keys `["nodes", "edges", "input_size", "output_size"]`

    (c) B. Hartl 2021
    """
    # prepare node data
    nodes = {k: {attr_k: attr_to_primitive(attr)
                 for attr_k, attr in g.nodes[k].items()
                 if attr_k in SERIALIZE_NODE_ATTRS}
             for k in g.nodes}

    # prepare edge data
    edges = nx.to_dict_of_dicts(g)
    drop_u = []
    for u, u_edges in edges.items():
        if u_edges == {}:
            drop_u.append(u)
            continue

        for v in u_edges.keys():
            u_edges[v] = {attr: attr_to_primitive(value)
                          for attr, value in u_edges[v].items()
                          if attr in SERIALIZE_EDGE_ATTRS}

    for u in drop_u:
        del edges[u]

    return dict(nodes=nodes, edges=edges, input_size=input_size, output_size=output_size)


def digraph_to_layers(g, input_layer, output_layer, verbose=False, zeros=None):
    """ construct a feedforward layer wise neural wired_rnn with recurrent options

    :param g: networkx DiGraph object
    :param input_layer: input layer keys
    :param output_layer: output layer keys
    :param verbose: boolean controlling whether print status reports should be given
    :param zeros: method to create zero-valued tensors of the used framework (defaults to numpy.zeros)
    :returns: OrderedDict with layer-key, layer-layout pairs

    (c) B. Hartl 2021
    """

    input_nodes = input_layer
    output_nodes = output_layer

    if zeros is None:
        from numpy import zeros

    forward, backwards = split_cyclic(g, input_nodes=input_nodes, output_nodes=output_nodes)
    forward_path = [fi for fi in nx.topological_sort(forward) if fi not in input_nodes and (g.succ[fi] or g.pred[fi])]

    try:
        recurrent_sources = [e[0] for e in backwards.edges]
        recurrent_destinations = [e[1] for e in backwards.edges]
    except (AttributeError, TypeError):
        recurrent_sources, recurrent_destinations = [], []

    if verbose:
        print('input nodes:', input_nodes)
        print('forward path:', forward_path)
        print('recurrent source:', recurrent_sources)
        print('recurrent destination:', recurrent_destinations)
        print('output nodes:', output_nodes)

    # forward path for now
    layers = []
    input_layer = list(input_nodes)

    for output_node in forward_path:

        # define weight matrix
        shape = (len(input_layer), 1)
        weights = zeros(shape)
        mask = (weights != 0.)  # all false by default

        # perform forward routing
        for i, in_node in enumerate(input_layer):
            edge_data = g.get_edge_data(in_node, output_node)
            if edge_data is None:
                continue
            weights[i, 0] = edge_data['weight']
            mask[i, 0] = True  # m wired connections

        # check for recurrent connections
        backward_nodes = []
        for source, dest in zip(recurrent_sources, recurrent_destinations):
            if dest == output_node:
                backward_nodes.append(source)
        is_hidden = output_node in recurrent_sources

        backward_weights = None
        backwards_mask = None
        if backward_nodes:
            # perform recurrent routing (assume, recurrent input is concatenated)
            backward_weights = zeros((len(backward_nodes), 1))
            backwards_mask = (backward_weights != 0.)

            for i, backward_source in enumerate(backward_nodes):
                edge_data = backwards.get_edge_data(backward_source, output_node)
                if edge_data is None:
                    continue
                backward_weights[i, 0] = edge_data['weight']
                backwards_mask[i, 0] = True

        # make layer layout
        layer = dict(nodes=[output_node],
                     shape=shape,
                     forward=weights,
                     mask=mask,
                     backward_nodes=backward_nodes,                          # recurrent input
                     backward_shape=getattr(backward_weights, 'shape', ()),  # recurrent shape
                     backwards=backward_weights,     # recurrent weight matrix
                     backwards_mask=backwards_mask,
                     hidden=is_hidden,               # boolean if has recurrent output
                     )

        special_attr = [k for k in layer.keys()]
        for attr_key, attr in forward.nodes[output_node].items():
            if attr_key not in special_attr:
                layer[attr_key] = attr

        layers.append(layer)

        # the next layer gets the following concatenated input:
        # - input from the previous layer (without recurrent connections)
        # - and the output from the current layer
        # - potential hidden state from previous stage
        input_layer += [output_node]

    # route concatenated output to output layer
    output_layer = input_layer
    output_nodes = list(output_nodes if hasattr(output_nodes, '__iter__') else [output_nodes])
    shape = len(output_layer), len(output_nodes)
    weights = zeros(shape)
    mask = (weights == 0.)
    for i, output_node in enumerate(output_nodes):
        output_source = output_layer.index(output_node)
        weights[output_source, i] = 1.
    layer = dict(nodes=output_nodes,
                 shape=shape,
                 forward=weights,
                 mask=mask,
                 node_attrs=[nx.get_node_attributes(forward, node) for node in output_nodes],
                 backward_nodes=[],
                 backward_indices=None,
                 backwards=None,
                 backwards_mask=None,
                 hidden=False,
                 )

    special_attr = [k for k in layer.keys()]
    for output_node in output_nodes:
        for attr_key, attr in forward.nodes[output_node].items():
            if attr_key not in special_attr:
                attr_list = layer.get(attr_key, [])
                attr_list.append(attr)
                layer[attr_key] = attr_list

    layers.append(layer)

    # wrap layers into ordered dict
    from collections import OrderedDict
    layers_dict = OrderedDict()
    for layer in layers[:-1]:

        # evaluate backward_nodes by layer indices
        layer['backward_indices'] = [source_index
                                     for backwards_node in layer['backward_nodes']             # first iteration
                                     for source_index, source_layer in enumerate(layers[:-1])  # second iteration
                                     if source_layer['nodes'][0] == backwards_node
                                     ]

        layers_dict[layer['nodes'][0]] = layer
    layers_dict['output'] = layers[-1]

    return layers_dict


class Wiring(object):
    """ A wrapper for a GNN that can be used for `NEATMethod` adaptation in the `mindcraft` framework

    (c) B. Hartl 2021
    """
    WIRING_KEYS = ['keys', 'n', 'bias', 'response', 'to_hidden', 'to_output',
                   'aggregation', 'activation', 'index_from_key']

    GRAPHVIZ_NODE_ATTRS = {'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'}
    GRAPHVIZ_INPUT_ATTRS = {'style': 'filled', 'shape': 'box', 'fillcolor': 'lightgray'}
    GRAPHVIZ_HIDDEN_ATTRS = {'style': 'filled', 'fillcolor': 'white'}
    GRAPHVIZ_OUTPUT_ATTRS = {'style': 'filled', 'fillcolor': 'lightblue'}

    TRUNCATE_FORWARD = 63
    """Truncate forward evaluation values to `2**6-1`"""

    def __init__(self, input_units, hidden_units, output_units):
        self.input = {k: input_units[k] for k in Wiring.WIRING_KEYS if k in input_units}
        self.hidden = {k: hidden_units[k] for k in Wiring.WIRING_KEYS if k in hidden_units}
        self.output = {k: output_units[k] for k in Wiring.WIRING_KEYS if k in output_units}

        self._forward_graph = None
        self._forward_path = None
        self._backwards_graph = None

        self.aggregations = {'sum': lambda x: np_sum(x, axis=-1),
                             'prod': lambda x: np_prod(x, axis=-1)}

        self.activations = {'linear': lambda x: x,
                            'sigmoid': lambda x: 1./(1.+np_exp(-x))}

    def n(self, key):
        units = getattr(self, key)

        try:
            return units['n']
        except KeyError:
            units['n'] = len(units['keys'])
            return units['n']

    def to_hidden(self, key):
        return getattr(self, key)['to_hidden']

    def to_output(self, key):
        return getattr(self, key)['to_output']

    def response(self, key):
        return getattr(self, key)['response']

    def bias(self, key):
        return getattr(self, key)['bias']

    def aggregation(self, key):
        return getattr(self, key)['aggregation']

    def activation(self, key):
        return getattr(self, key)['activation']

    def keys(self, key):
        return getattr(self, key)['keys']

    def indices(self, key):
        return [getattr(self, key)['index_from_key'][k] for k in self.keys(key)]

    def node_attributes(self, key):
        attributes = {}

        try:
            attributes['key'] = self.keys(key)
        except KeyError:
            pass

        try:
            attributes['index'] = self.indices(key)
        except KeyError:
            pass

        try:
            attributes['bias'] = self.bias(key)
        except KeyError:
            pass

        try:
            attributes['response'] = self.response(key)
        except KeyError:
            pass

        try:
            attributes['aggregation'] = self.aggregation(key)
        except KeyError:
            pass

        try:
            attributes['activation'] = self.activation(key)
        except KeyError:
            pass

        return attributes

    @property
    def digraph(self):
        g = DiGraph(directed=True)

        for graph_attributes in [self.node_attributes('input'),
                                 self.node_attributes('hidden'),
                                 self.node_attributes('output')]:
            graph_keys = graph_attributes.pop('key')
            for i, node_key in enumerate(graph_keys):
                node_attr = {k: v[i] for k, v in graph_attributes.items()}
                g.add_node(node_key, **node_attr)

        for (conn, in_keys, out_keys) in [(self.to_hidden('input'), self.input['keys'], self.hidden['keys']),
                                          (self.to_output('input'), self.input['keys'], self.output['keys']),
                                          (self.to_output('hidden'), self.hidden['keys'], self.output['keys']),
                                          (self.to_hidden('hidden'), self.hidden['keys'], self.hidden['keys']),
                                          (self.to_hidden('output'), self.output['keys'], self.hidden['keys']),
                                          (self.to_output('output'), self.output['keys'], self.output['keys'])]:
            for pairs, weight in zip(*conn):
                input_idx, output_idx = pairs
                input_key, output_key = in_keys[input_idx], out_keys[output_idx]
                g.add_edge(input_key, output_key, weight=weight, enabled=True)

        return g

    # def draw(self, node_names=None, node_colors=None, fmt='svg', filename=None, view=True):
    #
    #     import graphviz
    #     dot = graphviz.Digraph(format=fmt, node_attr=self.GRAPHVIZ_NODE_ATTRS)
    #
    #     if node_names is None:
    #         node_names = {}
    #
    #     assert type(node_names) is dict
    #
    #     if node_colors is None:
    #         node_colors = {}
    #
    #     assert type(node_colors) is dict
    #
    #     for keys, attrs in [(self.input['keys'], self.GRAPHVIZ_INPUT_ATTRS),
    #                         (self.output['keys'], self.GRAPHVIZ_OUTPUT_ATTRS),
    #                         (self.hidden['keys'], self.GRAPHVIZ_HIDDEN_ATTRS)]:
    #         for k in keys:
    #             name = node_names.get(k, str(k))
    #             node_attrs = {k: v for k, v in attrs.items()}
    #             if k in node_colors:
    #                 node_attrs['fillcolor'] = k
    #
    #             dot.node(name, _attributes=node_attrs)
    #
    #     # connections
    #     for (conn, in_keys, out_keys) in [(self.to_hidden('input'), self.input['keys'], self.hidden['keys']),
    #                                       (self.to_output('input'), self.input['keys'], self.output['keys']),
    #                                       (self.to_output('hidden'), self.hidden['keys'], self.output['keys']),
    #                                       (self.to_hidden('hidden'), self.hidden['keys'], self.hidden['keys']),
    #                                       (self.to_hidden('output'), self.output['keys'], self.hidden['keys']),
    #                                       (self.to_output('output'), self.output['keys'], self.output['keys'])]:
    #         for pairs, weight in zip(*conn):
    #             input_idx, output_idx = pairs
    #             input_key, output_key = in_keys[input_idx], out_keys[output_idx]
    #
    #             a = node_names.get(input_key, str(input_key))
    #             b = node_names.get(output_key, str(output_key))
    #             # style = 'solid' if cg.enabled else 'dotted'
    #             color = 'green' if weight > 0 else 'red'
    #             width = str(0.25 + abs(weight / 5.0))
    #             dot.edge(a, b, _attributes={'style': 'solid', 'color': color, 'penwidth': width})
    #
    #     if filename is not None:
    #         import os
    #         if not os.path.exists(os.path.dirname(filename)):
    #             os.makedirs(os.path.dirname(filename))
    #
    #         dot.render(filename, view=view)
    #
    #     elif view:
    #         import os
    #         tmp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
    #         dot.view(cleanup=True, filename=tmp_file)
    #
    #     return dot

    def get_aggregation_function(self, node):
        if isinstance(node, dict):
            if node['index'] in self.aggregations:
                return self.aggregations[node['index']]

            node = node['aggregation']

        try:
            if isinstance(node, str):
                return getattr(self.aggregations, node)

            return node

        except AttributeError:
            return None

    def get_activation_function(self, node):
        if isinstance(node, dict):
            if node['index'] in self.activations:
                return self.activations[node['index']]

            node = node['activation']

        try:
            if isinstance(node, str):
                return getattr(self.activations, node)

            return node

        except AttributeError:
            return None

    @property
    def has_hidden(self):
        return self.n('hidden') > 0

    @classmethod
    def parse(cls, g: Union[DiGraph, DiGraph], input_size, output_size, prune_unused: bool = False):
        """ parse tensor-transformable wiring information from nx-graph object

        this also works for other frameworks, not only pytorch.
        """

        # parse genome into input, hidden, output
        input_units, hidden_units, output_units = {}, {}, {}

        nodes = g.nodes
        node_keys = sorted([node for node in g.nodes])

        # parse keys
        input_units['keys'] = sorted([k for k in node_keys if k < 0])
        output_units['keys'] = sorted([k for k in node_keys if 0 <= k < output_size])
        assert len(input_units['keys']) == input_size
        assert len(output_units['keys']) == output_size

        hidden_units['keys'] = sorted([k for k in node_keys
                                       if k not in input_units['keys'] and k not in output_units['keys']])

        # parse responses
        hidden_units['response'] = [nodes[k]['response'] for k in hidden_units['keys']]
        output_units['response'] = [nodes[k]['response'] for k in output_units['keys']]

        # parse aggregation
        hidden_units['aggregation'] = [nodes[k]['aggregation'] for k in hidden_units['keys']]
        output_units['aggregation'] = [nodes[k]['aggregation'] for k in output_units['keys']]

        # parse activation
        hidden_units['activation'] = [nodes[k]['activation'] for k in hidden_units['keys']]
        output_units['activation'] = [nodes[k]['activation'] for k in output_units['keys']]

        # parse biases
        hidden_units['bias'] = [nodes[k]['bias'] for k in hidden_units['keys']]
        output_units['bias'] = [nodes[k]['bias'] for k in output_units['keys']]

        # simplification
        required_keys = required_for_output(input_units['keys'], output_units['keys'], (e for e in g.edges))

        if prune_unused:
            connections = set()
            for cg in g.edges:
                edge_attr = get_nested_attr(g, cg)
                if edge_attr['enabled']:
                    connections.add(cg)

            used_nodes = set(output_units['keys'])
            pending = set(output_units['keys'])
            while pending:
                new_pending = set()
                for a, b in connections:
                    if b in pending and a not in used_nodes:
                        new_pending.add(a)
                        used_nodes.add(a)
                pending = new_pending

            for i, k in enumerate(output_units['keys']):
                if k not in used_nodes:
                    output_units['bias'][i] = 0.

            keys_to_del = []
            for i, k in enumerate(hidden_units['keys']):
                wired = False
                for conn in g.edges:
                    if k in conn:
                        wired = True
                        break

                if not wired:
                    keys_to_del.append(i)

            for v in hidden_units.values():
                for i in reversed(sorted(keys_to_del)):
                    del v[i]

        else:
            used_nodes = node_keys

        # connectivity mapping
        for genome_units in [input_units, hidden_units, output_units]:
            genome_units['n'] = len(genome_units['keys'])
            genome_units['index_from_key'] = {k: i for i, k in enumerate(genome_units['keys'])}

        # perform wiring,  tuples of indices and weights ([(idx_in, idx_out)], [weight])
        input_units['to_hidden'], input_units['to_output'] = ([], []), ([], [])
        hidden_units['to_hidden'], hidden_units['to_output'] = ([], []), ([], [])
        output_units['to_hidden'], output_units['to_output'] = ([], []), ([], [])

        for conn in g.edges:
            conn_attr = get_nested_attr(g, conn)
            if not conn_attr['enabled']:  # dead connection
                continue

            key_in, key_out = conn
            if prune_unused:
                if key_in not in required_keys and key_out not in required_keys:  # dead connection
                    continue

                if key_in not in used_nodes:  # dead connection
                    continue

            # identify wiring of connection through input, hidden and output units
            wired = False
            for in_units in [input_units, hidden_units, output_units]:
                if key_in not in in_units['keys']:  # input unit node not in current input-unit-collection
                    continue

                for out_units, wiring in [(hidden_units, 'to_hidden'), (output_units, 'to_output')]:
                    if key_out not in out_units['keys']:  # output unit node not in current output-unit-collection
                        continue

                    idx_in = in_units['index_from_key'][key_in]
                    idx_out = out_units['index_from_key'][key_out]

                    idx_wiring, weight_wiring = in_units[wiring]  # get corresponding wiring tuple
                    idx_wiring.append((idx_in, idx_out))  # route index pairs
                    weight_wiring.append(conn_attr['weight'])  # set corresponding weight

                    wired = True
                    break
                break

            if not wired:
                raise IndexError(f'No connection between keys {key_in} and {key_out}.')

        return cls(input_units=input_units, hidden_units=hidden_units, output_units=output_units)

    def to_dict(self):
        return digraph_to_dict(self.digraph, self.input['n'], self.output['n'])

    def __eq__(self, other):
        # transform to dicts
        g = self.to_dict()
        o = other.to_dict() if isinstance(other, Wiring) else other

        # check base-level keys
        pending = set(o.keys())
        for key in g.keys():
            if key not in pending:
                return False

            pending.remove(key)

        if pending:
            return False

        # check input / output size
        for p in ['input_size', 'input_size']:
            if g[p] != o[p]:
                return False

        # check nodes
        pending = set(o['nodes'].keys())
        for key, attrs in g['nodes'].items():
            if key not in pending:
                return False

            if attrs != o['nodes'][key]:
                return False

            pending.remove(key)

        if pending:
            return False

        # check edges
        pending_u = set(o['edges'].keys())
        for u, edges_u in g['edges'].items():
            if u not in pending_u:
                return False

            pending_v = set(o['edges'][u].keys())
            for v, attrs in edges_u.items():
                if v not in pending_v:
                    return False

                if attrs != o['edges'][u][v]:
                    return False

                pending_v.remove(v)

            pending_u.remove(u)

        if pending_u:
            return False

        return True

    @classmethod
    def parse_neat(cls, genome, config, **kwargs):
        g, (input_size, output_size) = neat_to_digraph(genome, config)

        kwargs['input_size'] = input_size
        kwargs['output_size'] = output_size
        return cls.parse(g=g, **kwargs)

    @classmethod
    def make(cls, wiring: Union[Graph, DiGraph, dict, 'Wiring'], *args, **kwargs):
        if isinstance(wiring, cls):
            return wiring

        if isinstance(wiring, (Graph, DiGraph)):
            return cls.parse(wiring, *args, **kwargs)

        if isinstance(wiring, dict):
            if not any(isinstance(a, (dict, Config)) for a in args):
                if not any(isinstance(a, (dict, Config)) for a in kwargs.values()):
                    # 1) try to extract input and output size
                    try:
                        if 'input_size' in kwargs:
                            input_size = kwargs.pop('input_size')
                        else:
                            input_size, *args = args

                    except ValueError:  # if no input_size in positional args
                        input_size = None

                    try:
                        if 'output_size' in kwargs:
                            output_size = kwargs.pop('output_size')
                        else:
                            output_size, *args = args

                    except ValueError:  # if no output_size in positional args
                        output_size = None

                    # 2) parse dict
                    wiring, (input_size, output_size) = dict_to_digraph(wiring,
                                                                        input_size=input_size,
                                                                        output_size=output_size)

                    # 3) create Wiring object
                    kwargs['input_size'] = input_size
                    kwargs['output_size'] = output_size
                    return cls.make(wiring, **kwargs)

        return cls.parse_neat(wiring, *args, **kwargs)

    def _backwards(self, states):
        return states

    def state_index(self, key: Union[int, str, list, tuple]):
        if hasattr(key, '__iter__') and not isinstance(key, str):
            return [self.state_index(k) for k in key]

        return self._forward_path.index(key)

    def forward(self, x, states=None, zeros=np_zeros, stack=np_stack, axis_kw='axis', **kwargs):
        """ Evaluate Wiring-graph, node-by-node, considering forward and backward (recurrent) states.

        :param x: input tensor
        :param states: recurrent (hidden) state information
        :param zeros: method to initialize zero-valued tensor
        :param stack: method to stack tensors along new axis
        :param axis_kw: keyword to address axis/dim argument in cat and stack methods
        :param kwargs: forwarded to nodes_to_layers
        :returns: tuple of (i) output of forward or time-series evaluation and (ii) hidden states (which default to 0)
        """
        requires_states = False
        if self._forward_graph is None:
            g = self.digraph
            forward, backwards = split_cyclic(g, input_nodes=self.input['keys'], output_nodes=self.output['keys'])
            forward_path = [fi for fi in nx.topological_sort(forward)
                            if fi not in self.input['keys'] and (g.succ[fi] or g.pred[fi] or fi in self.output['keys'])]

            self._forward_graph = forward
            self._forward_path = forward_path  # chronological node evaluation
            self._backwards_graph = backwards  # possible recurrent graph with backward loops
            requires_states = self._backwards_graph is not None

        graph_eval = zeros(x.shape[0], len(self._forward_path))  # last hidden state of time-series
        if states is None and requires_states:
            states = zeros(graph_eval.shape)

        y = []  # return value for each potential time-step
        time_series = len(x.shape) > 2  # boolean, specifying whether time-series is provided
        for t in range(1 if not time_series else x.shape[1]):  # iterate over (possible) time-steps
            # Pick correct sequential input, in case of time-series
            x_t = x if not time_series else x[:, t]

            # Process forward path, node by node -> fills `graph_eval[:, id]` with values
            [self._forward(x=x_t, graph_eval=graph_eval, states=states, node=node, id=id, stack=stack, axis_kw=axis_kw)
             for id, node in enumerate(self._forward_path)]

            # Process recurrent feedback state (e.g., with LTC mechanism ... WIP)
            graph_eval = self._backwards(graph_eval)

            # Append evaluation of final layer (output layer)
            y.append(graph_eval[:, self.output['keys']])

        if not time_series:
            return y[-1], graph_eval

        return stack(y, **{axis_kw: 1}), graph_eval

    def _forward(self, x, graph_eval, states, node, id, stack, axis_kw):
        y = []  # aggregation list
        for graph, values in [(self._forward_graph, graph_eval),  # forward path, evaluations filled subsequently
                              (self._backwards_graph, states)]:    # backwards path, states from prior evaluations

            if graph is None:
                continue

            if node not in graph.nodes:
                continue

            for key in graph.pred[node]:
                edge = graph[key][node]
                if not edge['enabled']:
                    continue

                try:  # check, whether node has been evaluated (if so, it has a state_index and is stored in values)
                    x_i = values[:, self.state_index(key)]
                except (ValueError, IndexError):  # assume input node keys [-1, ..., -n]_NEAT -> [0, ..., n-1]_x
                    x_i = x[:, -key-1]

                y.append(edge['weight'] * x_i)

        node = self._forward_graph.nodes[node]
        aggregation = self.get_aggregation_function(node)
        activation = self.get_activation_function(node)

        z = node['bias']
        if len(y) != 0:
            y = stack(y, **{axis_kw: -1})

            try:  # AGGREGATION FUNCTION, MAY IMPLEMENT `response`-SPECIFIC BEHAVIOUR
                assert aggregation is not None
                z = partial(aggregation, response=node['response'])(y) + z
            except TypeError:
                z = aggregation(y) * node['response'] + z
            except AssertionError:
                pass

        try:  # ACTIVATION FUNCTION, MAY IMPLEMENT `response`-SPECIFIC BEHAVIOUR
            assert activation is not None
            z = partial(activation, response=node['response'])(z)
        except TypeError:
            z = activation(z)
        except AssertionError:
            z = z.sum(**{axis_kw: -1})

        return self._put_forward(value=z.flatten(), graph_eval=graph_eval, id=id)

    def _put_forward(self, value, graph_eval, id):
        # regulate output -> exploding values possible
        value = space_clip(value, (-self.TRUNCATE_FORWARD, self.TRUNCATE_FORWARD))

        # store forward evaluation in dynamic variable, and return
        graph_eval[:, id] = value
        return value
