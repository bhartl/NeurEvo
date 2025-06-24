from mindcraft.torch.module import Patchwork
from torch.distributions import Normal, MixtureSameFamily, Categorical
from torch import Tensor, ones, exp, cdist, eye
from torch.nn import Parameter


class GaussianProcess(Patchwork):
    REPR_FIELDS = ("input_size", "output_size", "distance_foo", "eps_x", *Patchwork.REPR_FIELDS)

    def __init__(self, input_size, output_size, distance_foo=None, delta_x0=1., delta_y0=1., eps_x=1e-6,
                 **patchwork_kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.distance_foo = distance_foo
        self.eps_x = eps_x

        # parameters
        self.delta_x = delta_x0
        self.delta_y = delta_y0
        self.noise = None

        patchwork_kwargs['omit_default'] = patchwork_kwargs.get('omit_default', True)
        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self):
        delta_x0, delta_y0 = self.delta_x, self.delta_y
        self.delta_x = Parameter(ones(self.input_size) * delta_x0)
        self.delta_y = Parameter(ones(self.input_size) * delta_y0)
        self.noise = Parameter(ones(self.input_size) * delta_y0)

    def kernel(self, x, x_prime):
        pairwise_distance = self.pairwise_distance(x, x_prime)
        return (self.delta_y**2) * exp(-pairwise_distance / (2. * (self.delta_x**2 + self.eps_x)))

    def covariance_matrix(self, x, x_prime):
        return self.kernel(x, x_prime) + eye(self.input_size) * self.noise

    def mean(self):
        return 0.

    def pairwise_distance(self, x1, x2):
        if self.distance_foo is None:
            return cdist(x1, x2, p=2)**2

        raise NotImplementedError(f"distance_foo {self.distance_foo}")

    def forward(self, x, *args: Tensor) -> Tensor:
        pass

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """ total loss function of the GP

        :param x: Data points tensor.
        :param y: Function values tensor.
        """

        pass


if __name__ == '__main__':
    import torch
    x = torch.randn(100, 10)
    gp = GaussianProcess(input_size=10, )
    gp.kernel(x)
