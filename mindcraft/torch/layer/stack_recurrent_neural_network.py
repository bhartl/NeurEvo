from torch.nn import Module, Conv1d, ModuleList
from torch import zeros_like, stack, concatenate


class StackRNN(Module):
    """ A low-key Stack-based Recurrent Neural Network implementation.

        The module maintains a fixed-size stack of previous input-tensors of model-calls to handle sequence data.
    """
    def __init__(self, input_size, hidden_size=3, num_layers=1,
                 bias=True, device=None, detached_stack=True, groups=1,
                 activation=None, batch_first=True, return_sequence=True):
        """ Constructs a STackRNN instance

        :param input_size: The number of input features (data-dim).
        :param hidden_size: The fixed stack-size, i.e., the rolling window of stored sequence data, defaults to 3.
        :param num_layers: The number of successive (`Conv1D`) layers that forward the stacked sequence data,
                           defaults to 1. Note that each layer has its own stack.
        :param bias: Boolean flag specifying whether to use bias or not in the layers, defaults to True.
        :param groups: Groups argument of the `Conv1d` layer, defaults to 1.
        :param activation: Optional activation function or list of activation functions (None is also allowed as list
                           entry), defaults to None.
        :param batch_first: Boolean flag specifying whether the batch- (if True) or the sequence dimension (if False)
                            is the first dimension of the input data, defaults to True.
        :param return_sequence: Boolean flag to return entire the sequences of output data corresponding
                                to the input sequence, defaults to True.
        :param detached_stack: Boolean flag to disable gradients in the stacked sequence data, defaults to True.
        :param device: The device to generate the model on ('cpu', 'cuda:n', defaults to None)
        """
        Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.groups = groups
        self.activation = activation
        self.detached_stack = detached_stack
        self.batch_first = batch_first
        self.return_sequence = return_sequence
        self._device = device

        self.stack = None
        self.layers = ModuleList()
        self._build()
        self.reset()

    def _build(self):
        for layer in range(self.num_layers):
            conv = Conv1d(in_channels=self.hidden_size,
                          out_channels=self.input_size,
                          kernel_size=self.input_size,
                          bias=self.bias,
                          device=self._device,
                          groups=self.groups,
                          )
            self.layers.append(conv)

        from mindcraft.torch.util import get_torch_layer
        if isinstance(self.activation, str):
            self.activation = get_torch_layer(self.activation)()
        elif hasattr(self.activation, '__iter__'):
            self.activation = [get_torch_layer(a)() for a in self.activation]

    def reset(self):
        self.stack = {layer: [] for layer in range(self.hidden_size)}

    def forward(self, x, states=None):
        """ Evaluates the model on the input x

        :param x: Either a 3 dimensional single sequence element or 4 dimensional sequence.
        :param states:
        :return: Tuple of model (output, states), where the output is either the entire sequence (if `x` is a sequence
                 and if `return_sequence` flag is set) or the last output sequence element.
        """
        is_sequence = len(x.shape) > 2
        seq_dim = int(self.batch_first)
        if not is_sequence:  # add sequence-dim either upfront (if not self.batch) or at dim 1
            x = x.unsqueeze(seq_dim)

        if self.batch_first:  # transform to (seq, batch, ...)
            x = x.transpose(0, 1)

        if states is not None:
            self.states = states

        y = []
        for xi in x:  # iterate over sequence data (to fill the stack successively)
            for layer_id, layer in enumerate(self.layers):  # iterate through the successive convolution layers
                # append the current sequence-element to the stack
                self.stack[layer_id].append(xi.detach().clone() if self.detached_stack else xi.clone())

                # make sure, the stack is initialized (filled with zeros if not)
                while not len(self.stack[layer_id]) == (self.hidden_size + 1):
                    self.stack[layer_id].insert(0, zeros_like(xi))

                # remove the oldest element and create tensor from stack
                self.stack[layer_id].pop(0)
                stack_tensor = stack(self.stack[layer_id], dim=1)

                # evaluate convolutional layer and corresponding call corresponding activation function
                xi = layer(stack_tensor).squeeze(-1)
                if self.activation is not None:
                    if hasattr(self.activation, '__iter__'):
                        if self.activation[layer_id]:
                            xi = self.activation[layer_id](xi)
                    else:
                        xi = self.activation(xi)

            # set output data (if no sequence was provided) or add output to sequence output
            y = xi if (not is_sequence or not self.return_sequence) else (y + [xi])

        if not is_sequence or not self.return_sequence:  # return last element
            return y, self.states

        # return entire sequence as tensor
        return stack(y, dim=seq_dim), self.states

    @property
    def states(self):
        """ return stack states in shape (NUM_LAYERS, BATCH_SIZE, INPUT_SIZE x HIDDEN_SIZE) """
        return stack([concatenate(self.stack[layer], dim=1) for layer in range(self.num_layers)])

    @states.setter
    def states(self, value):
        """ set stack states {LAYER_ID: list([(BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE)])} from
            value of shape (NUM_LAYERS, BATCH_SIZE, INPUT_SIZE x HIDDEN_SIZE) """
        self.stack = {i: [vi.reshape(-1, self.input_size, self.hidden_size)[:, :, h] for h in range(self.hidden_size)]
                      for i, vi in enumerate(value)}

    def train(self, mode=True):
        self.training = mode

        for layer in self.layers:
            layer.train(mode)

    def eval(self):
        self.train(False)
