""" PyTorch-adaptation of the aggregation functions of the original NEAT-python implementation

[1] K.O. Stanley, R Miikkulainen, Evolutionary Computation 10(2): 99-127 (2002)
[2] NEAT-python https://neat-python.readthedocs.io/en/latest/index.html

(c) B. Hartl 2021
"""

from torch import prod as torch_prod
from torch import sum as torch_sum
from torch import max as torch_max
from torch import min as torch_min
from torch import abs as torch_abs
from torch import full_like as torch_full_like
from torch import normal as torch_normal
from torch import median as torch_median
from torch import mean as torch_mean
from torch.nn.functional import relu as torch_relu
from torch.nn import Parameter as torch_parameter
from torch import Tensor
from typing import Union


def product(x: Tensor, dim=-1) -> Tensor:
    return torch_prod(x, dim=dim)


def sum(x: Tensor, dim=-1) -> Tensor:
    return torch_sum(x, dim=dim)


def max(x: Tensor, dim=-1) -> Tensor:
    return torch_max(x, dim=dim)[0]


def min(x: Tensor, dim=-1) -> Tensor:
    return torch_min(x, dim=dim)[0]


def maxabs(x: Tensor, dim=-1) -> Tensor:
    return torch_max(torch_abs(x), dim=dim)[0]


def median(x: Tensor, dim=-1) -> Tensor:
    return torch_median(x, dim=dim)[0]


def mean(x: Tensor, dim=-1) -> Tensor:
    return torch_mean(x, dim=dim)[0]


def dropout(x: Tensor, dim=-1, response: Union[Tensor, float] = 1.) -> Tensor:
    if not isinstance(response, Tensor):
        response = torch_full_like(x, response)

    elif isinstance(response, torch_parameter):
        response = torch_full_like(x, response.item())

    x = torch_relu(torch_normal(x, torch_abs(response)) - 1.2195121951219512)
    return sum(x, dim=dim)


class Aggregation(object):
    """ Basis class for aggregation function that has a memory

    (c) B. Hartl 2021
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, dim=-1):
        return sum(x, dim=dim)
