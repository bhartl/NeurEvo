from torch.nn import Module
from torch import sqrt, atan2, cos, sin, concatenate, no_grad, randn_like, pi, sign, abs, zeros_like
from mindcraft.torch.module import Patchwork
from mindcraft.torch.activation import get_activation_function


class Complexify(Patchwork):
    REPR_FIELDS = ("real", "complex_output", "activation", *Patchwork.REPR_FIELDS)

    def __init__(self, real: Patchwork, complex_output=True, activation=None, **patchwork_kwargs):
        Module.__init__(self)
        self.real = real
        self.imag = None
        self.complex_output = complex_output
        self.activation = activation
        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self):
        self.real = Patchwork.make(self.real)
        self.real.is_nested = True

        self.imag = self.real.copy()
        with no_grad():
            for imag_weight, real_weight in zip(self.imag.parameters(), self.real.parameters()):
                phase = randn_like(real_weight.data)
                imag_weight.data = real_weight.data * sin(phase)
                real_weight.data = real_weight.data * cos(phase)

        self.foo = get_activation_function(self.activation)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)

        return dict_repr

    def forward(self, x_real, x_imag=None):
        if x_imag is None:
            x_real, x_imag = self.split_real_imag(x_real)

        y_real = self.real(x_real) - self.imag(x_imag)  # + self.imag.bias
        y_imag = self.real(x_imag) + self.imag(x_real)  # - self.real.bias

        # Apply the activation function to the magnitude of h
        y_abs = self.get_abs(y_real, y_imag)
        if self.foo is not None:
            y_abs = self.foo(y_abs)

        if not self.complex_output:
            return y_abs

        # Correct for the phase of h using the phase of the real part
        phase = self.get_phase(y_real, y_imag)
        y_real, y_imag = self.split_real_imag(y_abs, phase)

        # Return the real and imaginary parts of the final output separately
        return y_real, y_imag

    @classmethod
    def get_abs(cls, x_real, x_imag):
        return sqrt(x_real**2 + x_imag**2)

    @classmethod
    def get_phase(cls, x_real, x_imag):
        return atan2(x_imag, x_real) % (2.*pi)

    @classmethod
    def split_real_imag(cls, abs_val, phase=None):
        if phase is None:
            # r, s = sqrt(abs(abs_val)), sign(abs_val)
            return abs_val, zeros_like(abs_val)  # r, s
        return abs_val * cos(phase), abs_val * sin(phase)

    @property
    def input_size(self):
        return self.real.input_size

    @property
    def output_size(self):
        return self.real.output_size

    @property
    def is_sequence_module(self):
        return self.real.is_sequence_module

    @property
    def states(self):
        if self.is_sequence_module and self.real.states is not None:
            # cat real and imag states: 2 x (num_layers, BS, SEQ, FEATURES) -> (num_layers x 2, BS, SEQ, FEATURES)
            states = tuple((concatenate([ri, ii], dim=0) for ri, ii in zip(self.real.states, self.imag.states)))
            return states
        return None

    @states.setter
    def states(self, value):
        if self.is_sequence_module:
            # split real and imag states: (num_layers x 2, BS, SEQ, FEATURES) -> 2 x (num_layers, BS, SEQ, FEATURES)
            n = self.real.num_layers
            self.real.states = tuple((v[:n] for v in value))
            self.imag.states = tuple((v[n:] for v in value))

    @property
    def num_layers(self):
        if self.is_sequence_module:
            return self.real.num_layers * 2  # real & imag
        return 0


if __name__ == '__main__':
    from mindcraft.torch.module import Recurrent, FeedForward
    from torch import randn
    rnn = Recurrent(input_size=10, hidden_size=2, output_size=5, num_layers=2)
    ff = FeedForward(input_size=5, output_size=5)

    # Use the Complexify decorator to create a complex-valued version of the layer
    complex_rnn = Complexify(rnn)
    complex_ff = Complexify(ff)

    # Test the complex-valued layer
    x_real = randn(32, 10)
    h_real, h_imag = complex_rnn(x_real, None)
    h_real, h_imag = complex_ff(h_real, h_imag)
    print(h_real.min(), h_real.max(), h_real.shape)
    print(h_imag.min(), h_imag.max(), h_imag.shape)
    print()
    print(Complexify.get_phase(h_real, h_imag))
    print()
    print(Complexify.get_abs(h_real, h_imag))
