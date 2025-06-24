from torch import tensor, float32, stack, zeros, tanh, concatenate
from torch.nn import Module, Linear, Sigmoid, RNN, Parameter, Identity
from typing import Union


class GRN(Module):
    """ A PyTorch implementation of a simple Gene-Regulatory-Network

    .. math::

        g_{t+1} = (1 - \\tau_2) \\times g_{t} + \\tau_1 \\sigma(W \\times g_{t})
    """
    def __init__(self,
                 num_genes,
                 tau_1: Union[float, Module] = 1.,
                 tau_2: Union[float, Module] = 0.2,
                 tau_as_param: bool = False,
                 bias=False,
                 device=None,
                 num_iterations=1,
                 activation=Sigmoid):
        """ Constructs a GRN instance

        :param num_genes: The number of input features (data-dim).
        :param tau_1: Float (or PyTorch Module evaluated on the genome input) defining the first time constant of the
                      model, defaults to 1.
        :param tau_2: Float (or PyTorch Module evaluated on the genome input) defining the second time constant of the
                      model, defaults to 0.2
        :param tau_as_param:
        :param bias: Boolean flag specifying whether to use bias or not in the layers, defaults to False.
        :param activation: Optional activation function or name of activation function (i.e., sigma),
                           defaults to `Sigmoid`.
        :param device: The device to generate the model on ('cpu', 'cuda:n', defaults to None)
        """
        Module.__init__(self)
        self.num_genes = num_genes
        self.bias = bias
        self.activation = activation
        self._device = device

        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_as_param = tau_as_param
        self.num_iterations = num_iterations

        self.gn = None
        self._build()

    def _build(self):
        self.gn = Linear(in_features=self.num_genes, out_features=self.num_genes,
                         bias=self.bias, device=self._device)

        if self.tau_as_param:
            self.tau_1 = Parameter(tensor(self.tau_1, device=self._device, dtype=float32))
            self.tau_2 = Parameter(tensor(self.tau_2, device=self._device, dtype=float32))

        from mindcraft.torch.util import get_torch_layer
        if isinstance(self.activation, str):
            self.activation = get_torch_layer(self.activation)
        if isinstance(self.activation, type):
            self.activation = self.activation()

    def forward(self, x, ):
        """ Evaluates the model on the input x

        :param x:
        :return:
        """

        tau_1, tau_2 = self.forward_tau(x)
        for _ in range(self.num_iterations - 1):
            x = self.forward_grn(x, tau_1=tau_1, tau_2=tau_2, detach=True)
        y = self.forward_grn(x, tau_1=tau_1, tau_2=tau_2)

        return y

    def forward_tau(self, x):
        tau_1 = self.tau_1
        if isinstance(tau_1, Module):
            tau_1 = tau_1(x)
            tau_1 = Sigmoid()(tau_1)

        tau_2 = self.tau_2
        if isinstance(tau_2, Module):
            tau_2 = tau_2(x)
            tau_2 = Sigmoid()(tau_2)

        if self.tau_as_param:
            tau_1 = Sigmoid()(tau_1)
            tau_2 = Sigmoid()(tau_2)

        return tau_1, tau_2

    def forward_grn(self, x, tau_1, tau_2, detach=False):
        y = tau_1 * self.activation(self.gn(x)) + (1. - tau_2) * x
        if detach:
            return y.detach()
        return y

    def to(self, device):
        if self.tau_as_param:
            self.tau_1 = Parameter(self.tau_1.to(device=device))
            self.tau_2 = Parameter(self.tau_2.to(device=device))
        return Module.to(self, device)


class G2RN(RNN):
    def __init__(self, *args,
                 tau_1: Union[float, Module] = 1.,
                 tau_2: Union[float, Module] = 0.2,
                 tau_as_param: bool = False,
                 bias=False,
                 activation=Sigmoid,
                 num_iterations=1,
                 device=None,
                 output_size=None,
                 **kwargs):
        RNN.__init__(self, *args, device=device, **kwargs)
        self.num_iterations = num_iterations
        self.output_size = output_size
        self.grn = GRN(num_genes=self.hidden_size, tau_1=tau_1, tau_2=tau_2, bias=bias, activation=activation,
                       device=device, tau_as_param=tau_as_param, num_iterations=num_iterations)

        self.head = Identity()
        if output_size:
            self.head = Linear(in_features=self.hidden_size, out_features=self.output_size, bias=bias)

    def forward(self, input, hx=None):
        seq_dim = int(self.batch_first)
        y = []
        idx = [slice(None)] * len(input.shape)
        for i in range(input.shape[seq_dim]):
            idx[seq_dim] = i
            x = input[tuple(idx)].unsqueeze(seq_dim)  # (BS, 1, ...) or (1, BS, ...)
            yi, hx = RNN.forward(self, x, hx)
            hx = self.grn(hx)
            y.append(yi)

        y = concatenate(y, dim=seq_dim)
        return self.head(y), hx


class RGRN(Module):
    """ Recurrent Gene Regulatory Network, with forward function of the form

        update_i = tanh(U(x) + W(state_i))

        state_{i+1} = (1 - \\tau_1) state_i + \\tau_2 \\times update_i

        return Activation(V(state_{i+1}), state_{i+1}

        inspired by Liquid Time Constant Network
    """

    def __init__(self, input_size, hidden_size, output_size=None, tau_1=0.75, tau_2=0.25, num_iterations=4,
                 num_layers=1, inner_merge=False,
                 bias=True, batch_first=True, regulate_x=False, activation=None):
        """ Constructs an RGRN instance

        :param input_size: The feature input dimension.
        :param hidden_size: The size of the hidden state.
        :param output_size: Optional output size, if defined, the hidden state will be projected by a linear model to
                            the specified output size.
        :param tau_1: Weight to obtain previous state by a ratio of (1-tau_1), defaults to 0.75.
        :param tau_2: Weight to include new information (gated x, state combination, `U(x) + W(state)`)
                      by a ratio (tau_2), defaults to 0.25
        :param num_iterations: Number of iterations to perform the state update.
        :param num_layers: NotImplemented: Number of G2RN layer, must be 1.
        :param bias: Boolean whether to use bias in the input matrix, U, and in the optional output matrix V.
        :param batch_first: Boolean flat to specify whether the first data dimension is the batch dimension or the
                            sequence dimension (if False), defaults to True.
        :param regulate_x: Boolean flat to specify whether to perform the state_i update with a constant input
                           update U(x) + W(state_i) for `num_iterations`, defaults to False.
        :param activation: Optional activation function (callable) that is applied after the output matrix V,
                           defaults to None.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.inner_merge = inner_merge
        self.num_iterations = num_iterations
        if num_layers != 1:
            raise NotImplementedError(f"`num_layers != 1`: got {num_layers}.")
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.regulate_x = regulate_x
        self.activation = activation

        # Define the input-to-hidden and hidden-to-hidden weight matrices
        self.U = Linear(in_features=input_size, out_features=hidden_size, bias=self.bias)
        self.W = Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        # Define the output weight matrix
        if output_size:
            self.V = Linear(in_features=hidden_size, out_features=output_size, bias=self.bias)
        else:
            self.V = Identity()

    def forward(self, x, state=None):
        if self.batch_first:
            batch_size, seq_len, *_ = x.size()
            seq_dim = 1
        else:
            seq_len, batch_size, *_ = x.size()
            seq_dim = 0

        if state is None:
            state = zeros(batch_size, self.hidden_size, device=x.device)
        elif len(state.shape) == 3:
            state = state[0]  # remove num_layers dummy dimension

        # Iterate over time steps
        y = []
        seq_idx = [slice(None)] * len(x.shape)
        for t in range(seq_len):
            seq_idx[seq_dim] = t
            xt = x[seq_idx]
            state = self._forward_xt(xt, state)
            yt = self.V(state)
            if self.activation is not None:
                yt = self.activation(yt)
            y.append(yt)

        state = state.unsqueeze(0)  # add num_layers dummy dimension
        return stack(y, dim=seq_dim), state

    def _forward_xt(self, xt, state):
        hx = self._gate_xt(xt, state)
        for i in range(self.num_iterations):
            state = (1. - self.tau_1) * state + hx * self.tau_2
            if self.regulate_x and i < self.num_iterations - 1:
                hx = self._gate_xt(xt, state)
        return state

    def _gate_xt(self, xt, state):
        if self.inner_merge:
            return tanh(self.U(xt) + self.W(state))
        return self.U(xt) + tanh(self.W(state))
