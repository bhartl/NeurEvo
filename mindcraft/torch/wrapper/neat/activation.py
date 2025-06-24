""" PyTorch-adaptation of the activation functions of the original NEAT-python implementation

[1] K.O. Stanley, R Miikkulainen, Evolutionary Computation 10(2): 99-127 (2002)
[2] NEAT-python https://neat-python.readthedocs.io/en/latest/index.html

(c) B. Hartl 2021
"""
import numpy as np
import torch
from torch import sigmoid as torch_sigmoid
from torch import tanh as torch_tanh
from torch import abs as torch_abs
from torch import exp as torch_exp
from torch import sin as torch_sin
from torch import cos as torch_cos
from torch import log as torch_log
from torch import Tensor
from torch import normal as torch_normal
import torch.nn.functional as functional
from typing import Union


def sigmoid(z: Tensor) -> Tensor:
    return torch_sigmoid(z * 5.)


def tanh(z: Tensor) -> Tensor:
    return torch_tanh(z * 2.5)


def sin(z: Tensor) -> Tensor:
    return torch_sin(z * 5.)


def cos(z: Tensor) -> Tensor:
    return torch_cos(z * 5.)


def gauss(z: Tensor) -> Tensor:
    return torch_exp(-5. * (z**2))


def relu(z: Tensor) -> Tensor:
    return functional.relu(z)


def noisy_relu(z: Tensor, response: Union[torch.Tensor, float] = 0.41) -> Tensor:
    """ Noisy ReLU of the form `ReLU(z + normal(mu, sigma))`, equivalent to dropout

    :param z: Aggregated neural signals.
    :param response: Width of noisy relu, defaults to `0.41`, (mean of noisy relu, defaults to `-0.5`),
                     cf. Fig. 7a in https://arxiv.org/pdf/2003.03384.pdf.
    """
    if not isinstance(response, torch.Tensor):
        response = torch.full_like(z, response)

    elif isinstance(response, torch.nn.Parameter):
        response = torch.full_like(z, response.item())

    return functional.relu(torch_normal(z, torch.abs(response)) - 0.5)


def softplus(z: Tensor) -> Tensor:
    return 0.2 * torch_log(1. + exp(z))


def identity(z: Tensor) -> Tensor:
    return z


def clamped(z: Tensor, bound=1.):
    z[z < -bound] = -1.
    z[z > bound] = 1.
    return z


def inv(z: Tensor):
    z = 1./z
    z[z != z] = 0.             # handle div by zero
    z[z == float('inf')] = 0.  # handle overflows
    return z


def rect_inv(z: Tensor, response: Union[torch.Tensor, float] = 1.):
    if not isinstance(response, torch.Tensor):
        response = torch.full_like(z, response)

    elif isinstance(response, torch.nn.Parameter):
        response = torch.full_like(z, response.item())

    s = torch.sign(z) * torch.sign(response)
    z = torch.maximum(abs(z), abs(response))

    s[s == 0] = 1.
    z = s / z
    z[z != z] = 0.             # handle div by zero
    z[z == float('inf')] = 0.  # handle overflows

    return z


def rect_abs_inv(z: Tensor, response: Union[torch.Tensor, float] = 1.):
    if not isinstance(response, torch.Tensor):
        response = torch.full_like(z, response)

    elif isinstance(response, torch.nn.Parameter):
        response = torch.full_like(z, response.item())

    z = 1 / torch.maximum(abs(z), abs(response))
    z[z != z] = 0.             # handle div by zero
    z[z == float('inf')] = 0.  # handle overflows

    return z


def log(z: Tensor) -> Tensor:
    z = abs(z)
    z[z < 1e-7] = 1e-7
    return torch_log(z)


def exp(z: Tensor) -> Tensor:
    return torch_exp(z)


def abs(z: Tensor) -> Tensor:
    return torch_abs(z)


def hat(z: Tensor) -> Tensor:
    z = 1. - torch_abs(z)
    z[z < 0.] = 0.
    return z


def square(z: Tensor) -> Tensor:
    return z**2


def noisy_square(z: Tensor, response: Union[torch.Tensor, float] = 1.) -> Tensor:
    if not isinstance(response, torch.Tensor):
        response = torch.full_like(z, response)

    elif isinstance(response, torch.nn.Parameter):
        response = torch.full_like(z, response.item())

    noisy_z = torch_normal(z, torch.abs(response)) # noise = noisy_z - z -> noisy_z = z + noise
    return noisy_z * (2.*z - noisy_z)   # -> equiv to `(z + noise) * (z - noise) == (noisy_z) * (z - [noisy_z] - z)`


def cube(z: Tensor) -> Tensor:
    return z**3


class Activation(object):
    def __init__(self, **kwargs):
        self._props = kwargs

    def __call__(self, z, response=1.):
        return identity(z) * response


class Grad(Activation):
    """ Stateful activation function which returns the scaled finite difference between the current and previous call.

    The `response` variable is used as scale. Additionally, a `delta_t` may be provided to account for step-size.

    The initial value for z is considered to be 0.

    (c) B. Hartl 2021
    """
    def __init__(self, delta_t=1., **kwargs):
        """ Constructs a stateful `Grad` activation function

        :param delta_t: float to account for step-size, defaults to 1.
        :param kwargs: to be forwarded to the parental class constructor.
        """
        self.delta_t = delta_t
        Activation.__init__(self, **kwargs)

        # states
        self.prev_z = None

    def __call__(self, z, response=1.):
        if self.prev_z is None:
            self.prev_z = z.clone()
            return z * 0.

        z, self.prev_z = response * (z - self.prev_z) / self.delta_t, z.clone()
        return z


class Integ(Activation):
    """ Stateful activation function which returns an iteratively integrated signal followin the trapezoid rule of
        integration from a successive time-series input, i.e., the data is provided iteratively and the activation
        function keeps track of the state of the integration.

    The `response` variable is used as scale. Additionally, a `delta_t` may be provided to account for step-size.

    The initial value for z is considered to be 0.

    (c) B. Hartl 2021
    """

    def __init__(self, delta_t=1., max_value=63., **kwargs):
        """ Constructs a stateful `Integ` activation function

        :param delta_t: float to account for step-size, defaults to 1.
        :param max_value: numerical max-value for the internal integration memory, defaults to 63.
        :param kwargs: to be forwarded to the parental class constructor.
        """
        self.delta_t = delta_t
        self.max_value = max_value
        Activation.__init__(self, **kwargs)

        # states
        self.prev_z = None
        self.y = None

    def __call__(self, z, response=1.):
        if self.prev_z is None:
            self.prev_z = z.clone()
            self.y = torch.zeros_like(z)
            return self.y.clone()

        self.y, self.prev_z = self.y + 0.5 * (z + self.prev_z) * self.delta_t * response, z.clone()
        self.y[self.y.abs() > self.max_value] = self.max_value
        return self.y.clone()
