""" Module utilizing `PyTorch` activation functions for `mindcraft` purposes.

(c) B. Hartl 2021
"""
import torch
from torch.nn import functional


TWO_PI = 2. * (torch.acos(torch.zeros(1)) * 2)


def get_activation_function(activation: str):
    """ get a callable activation function from the torch framework or from the torch.nn.functional
    module by str representation. Per default and on error None is returned. """
    try:
        foo = getattr(torch, activation, getattr(torch.nn, activation, getattr(functional, activation, None)))
        if isinstance(foo, type):
            return foo()
        return foo
    except TypeError:
        return None


def call_activation(x, foo=None):
    """ return result of an activation function `foo` on the input `x`, return `x` if `foo is None` """
    if foo is None:
        return x

    return foo(x)


def perceptron(x, work_on_clone=True):
    """ returns clone of x with all elements > 0 set to 1 and 0 otherwise.

    :param x: input tensor
    :param work_on_clone: Boolean specifying whether to clone x or not. If falls, x is changed and returned.
    """
    y = torch.clone(x) if work_on_clone else x
    y[y < 0] = 0.
    y[y > 0] = 1.
    return y


def gaussian(x: torch.Tensor, mu=0., var=1.):
    # global TWO_PI
    phase = 0.5 * ((x - mu) ** 2) / var
    if not isinstance(phase, torch.Tensor):
        phase = torch.tensor(phase, device=x.device)

    try:
        return torch.exp(-phase) / torch.sqrt(TWO_PI * var)

    except RuntimeError:
        if x.device != TWO_PI.device:
            globals()['TWO_PI'] = TWO_PI.to(x.device)
            return gaussian(x=x, mu=mu, var=var)
        raise