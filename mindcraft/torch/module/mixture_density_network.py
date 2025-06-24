import torch
from mindcraft.torch.module import Patchwork
from mindcraft.torch.module import FeedForward


class MixtureDensityNetwork(FeedForward):
    """ A pytorch implementation of the Mixture Density Network [ Bishop, 1994 ]

    adapted from https://github.com/tonyduan/mixture-density-network under the MIT license """

    DIAGONAL_NOISE = "diag"
    ISOTROPIC_NOISE = "iso"
    CLUSTER_ISOTROPIC_NOISE = "ciso"
    NOISE_TYPE = (DIAGONAL_NOISE, ISOTROPIC_NOISE, CLUSTER_ISOTROPIC_NOISE)

    SAMPLE_FORWARD = 'sample'
    PARAMS_SAMPLE = 'params'
    FORWARD_MODE = (SAMPLE_FORWARD, PARAMS_SAMPLE)

    REPR_FIELDS = ("num_components", "eps", "mode", "noise", *FeedForward.REPR_FIELDS)

    def __init__(self, input_size, output_size, num_components, hidden_size=None,
                 activation='ReLU', dropout=None, batch_norm=None, bias=True,
                 eps=1e-6, mode=SAMPLE_FORWARD, noise=DIAGONAL_NOISE,
                 **kwargs):
        """ Constructs a `MixtureDensity` instance

        :param input_size: Integer specifying the number of input dimensions to the network.
        :param output_size: Integer specifying the number of output dimensions of the network.
        :param hidden_size: Optional integer or tuple/list of integers specifying the successive dimensions of hidden layers
                            of the network.
        :param activation: Optional representation or list of representations of PyTorch activation
                           functions (e.g. "ReLU", ...) that are element wise retrieved via
                           `mindcraft.torch.activation.get_activation_function(...)`. If a single activation name
                           is provided, all layers are activated with the same function. If several are provided,
                           each layer will have its own activation. Note that `None` is a valid option.
                           Defaults to None.
        :param bias: Optional boolean flag or list of boolean flag to either globally or layer-wise enable/disable
                     the use of biases. Defaults to True.
        :param batch_norm: Optional boolean flag or list of boolean flag to either globally or layer-wise enable/disable
                           the use of BatchNorm layers after the network application but prior to activation and dropout.
                           Note that `None` is a valid option, boolean values will be used as `affine` arguments in the
                           BatchNorm initializations, respectively. Defaults to None.
        :param dropout: Optional float or list of float values to either globally or layer-wise define the dropout
                        fraction after the network activation, defaults to None.
        :param num_components: Number of components in the mixture model
        :param noise:
        :param eps:
        """
        self.num_components = num_components
        self.mode = mode
        assert noise in self.NOISE_TYPE or isinstance(noise, float)
        self.noise = noise
        self.eps = eps

        self.ff_pi = None
        self.ff_normal = None
        FeedForward.__init__(self,
                             input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             activation=activation, dropout=dropout, batch_norm=batch_norm, bias=bias,
                             **kwargs)

    def _build(self):
        activation = self.activation
        if activation and isinstance(activation, str):
            activation = [activation] * self.num_hidden + [None]

        dropout = self.dropout
        if dropout and isinstance(dropout, int):
            dropout = [dropout] * self.num_hidden + [None]

        batch_norm = self.batch_norm
        if batch_norm and isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * self.num_hidden + [None]

        self.nn = None
        self.ff_pi = FeedForward(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=self.num_components,
                                 bias=self.bias,
                                 activation=activation,
                                 batch_norm=batch_norm,
                                 dropout=dropout, )
        self.ff_normal = FeedForward(input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     output_size=self.output_size * self.num_components + self.num_channels_sigma,
                                     bias=self.bias,
                                     activation=activation,
                                     batch_norm=batch_norm,
                                     dropout=dropout, )

    @property
    def num_hidden(self):
        num_hidden = 0
        if self.hidden_size:
            if hasattr(self.hidden_size, '__iter__'):
                num_hidden = len(self.hidden_size)
            else:
                num_hidden = 1

        return num_hidden

    @property
    def num_channels_sigma(self):
        if self.noise == self.DIAGONAL_NOISE:
            return self.output_size * self.num_components

        if self.noise == self.ISOTROPIC_NOISE:
            return self.num_components

        if self.noise == self.CLUSTER_ISOTROPIC_NOISE:
            return 1

        if isinstance(self.noise, float):  # fixed noise level
            return 0

        raise ValueError(self.noise)

    def forward(self, x, *args, mode=None):
        """

        :param x:
        :param mode:
        :returns: Tuple of (log_pi: (BS, num_components), mu: (BS, num_components, output_size), sigma: (BS, num_components, output_size))
        """
        mode = mode or self.mode
        if mode == self.SAMPLE_FORWARD:
            return self.sample(x, *args)
        x = Patchwork.forward(self, x, *args)

        log_pi = self.forward_pi(x)
        mu, sigma = self.forward_normal(x)
        return log_pi, mu, sigma

    def forward_pi(self, x):
        return torch.log_softmax(self.ff_pi(x), dim=-1)

    def forward_normal(self, x):
        normal_params = self.ff_normal(x)
        mu = normal_params[..., :self.output_size * self.num_components]
        sigma = normal_params[..., self.output_size * self.num_components:]
        if self.noise == self.DIAGONAL_NOISE:
            sigma = torch.exp(sigma + self.eps)

        elif self.noise == self.ISOTROPIC_NOISE:
            sigma = torch.exp(sigma + self.eps).repeat(1, self.output_size)

        elif self.noise == self.CLUSTER_ISOTROPIC_NOISE:
            sigma = torch.exp(sigma + self.eps).repeat(1, self.num_components * self.output_size)

        elif isinstance(self.noise, float):
            sigma = torch.full_like(mu, fill_value=self.noise)

        else:
            raise ValueError(self.noise)

        mu = mu.reshape(-1, self.num_components, self.output_size)
        sigma = sigma.reshape(-1, self.num_components, self.output_size)
        return mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x, mode=self.PARAMS_SAMPLE)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (-0.5 * torch.einsum("bij,bij->bi", z_score, z_score) - torch.sum(torch.log(sigma), dim=-1))
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x, *args):
        log_pi, mu, sigma = self.forward(x, *args, mode=self.PARAMS_SAMPLE)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.gather(rand_normal, index=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples

    def to(self, device):
        self.ff_pi = self.ff_pi.to(device)
        self.ff_normal = self.ff_normal.to(device)
        return Patchwork.to(self, device)
