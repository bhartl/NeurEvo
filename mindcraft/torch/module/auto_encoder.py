from torch import Tensor, tensor, exp, zeros_like, ones_like, stack, sum
from torch.distributions import Normal, MixtureSameFamily, Categorical
from mindcraft.torch.module import Patchwork
from torch.nn import MSELoss, Parameter
from typing import Optional, Tuple, Union
from torch.nn import Module, Softmax, ReLU


class AutoEncoder(Patchwork):
    """ PyTorch based Auto Encoder

    Depending on whether `mu` and `sigma` networks are defined, the AE can be instantiated as Variational Auto Encoder.

    Tries to reconstruct an input x -> (mu, log_var) ~ z -> x_hat

    - via an encoding step, predicting the mean (mu) and the confidence (log variance)
      of a datapoint x in the latent space
    - sampling a random number from the normal distribution (mu, sigma=exp(log_var/2))
    - reconstructing x from z through a decoder

    References:

    - see `deep generative learning <https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/>`_
    - see `towardsdatascience <https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch

    (c) B. Hartl 2021
    """

    REPR_FIELDS = (
            "encoder",
            "decoder",
            "mu",
            "log_var",
            "weight",
            "mu_prime",
            "log_var_prime",
            "weight_prime",
            "beta",
            "log_scale",
            "num_components",
            *Patchwork.REPR_FIELDS,
        )
    
    def __init__(self,
                 encoder: Patchwork,
                 decoder: Patchwork,
                 mu: Optional[Patchwork] = None,
                 log_var: Optional[Patchwork] = None,
                 weight: Optional[Patchwork] = None,
                 mu_prime: Optional[Patchwork] = None,
                 log_var_prime: Optional[Patchwork] = None,
                 weight_prime: Optional[Patchwork] = None,
                 beta: float = 1.,
                 log_scale: float = 0.,
                 num_components: int = 1,
                 **patchwork_kwargs,
                 ):
        """ Constructs a VAE instance

        :param encoder: Encoder instance or dict, specifying a `mindcraft.torch.module.Patchwork` argument.
                        In case of a regular AE (no `mu` and `log_var` networks), the `output_size` of the encoder
                        must match the `input_size` of the decoder. Otherwise, in case of a VAE, the `output_size`
                        of the `encoder` must match the `input_size` of both the `mu` and `log_var` networks,
                        and the `output_size` of the latter must match the `input_size` of the decoder.
        :param decoder: Decoder instance or dict, specifying a `mindcraft.torch.module.Patchwork` argument.
                        In case of a regular AE (no `mu` and `log_var` networks), the `output_size` of the encoder
                        must match the `input_size` of the decoder. Otherwise, in case of a VAE, the `output_size`
                        of the `encoder` must match the `input_size` of both the `mu` and `log_var` networks,
                        and the `output_size` of the latter must match the `input_size` of the decoder.
        :param mu: (Optional) encoder mean estimate instance or dict, specifying a `mindcraft.torch.module.Patchwork`
                   argument, defaults to None (regular AE).
                   In case of a VAE, the `output_size` of the `encoder` network must match the input  of both the
                   `mu` and `log_var` networks, and the `output_size` of the latter must match the `input_size`
                   of the decoder.
        :param log_var: (Optional) encoder `log_var` estimate instance or dict, specifying a
                        `mindcraft.torch.module.Patchwork`, defaults to None (regular AE).
                        In case of a VAE, the `output_size` of the `encoder` network must match the input  of both the
                        `mu` and `log_var` networks, and the `output_size` of the latter must match the `input_size`
                        of the decoder.
        :param weight: (Optional) encoder `log_weight` `mindcraft.torch.module.Patchwork` instance or representation
                       that weights the different PDFs of the Mixture Density Networks (in case `num_components > 1`),
                       defaults to None (regular AE). In case of an MDN-VAE, the `output_size` of the `encoder` network
                       must match the input  of both the `mu`, `log_var` and `weight` networks, and the `output_size`
                       of the latter must match the `input_size` of the decoder.
        :param beta: (Optional) Latent-loss weighting factor wrt. Reconstruction-loss in the loss function.
        :param log_scale: (Optional) Log scale in case of `gaussian_likelihood' reconstruction loss (defaults to 0.)
        :param num_components: (Optional) integer providing the number of Gaussian distributions - in a mixture
                               density network sense - that are used to sample latent space data.
        """
        Module.__init__(self)
        self.encoder = encoder
        self.mu = mu
        self.log_var = log_var
        self.weight = weight
        self.mu_prime = mu_prime
        self.log_var_prime = log_var_prime
        self.weight_prime = weight_prime
        self.decoder = decoder
        self.beta = beta
        self.log_scale = log_scale
        self._log_scale = None
        self.num_components = num_components

        patchwork_kwargs['omit_default'] = patchwork_kwargs.get('omit_default', True)
        Patchwork.__init__(self, **patchwork_kwargs)

    @property
    def is_variational(self):
        return self.mu is not None and self.log_var is not None

    @property
    def latent_size(self):
        if self.is_variational:
            return self.mu.output_size

        return self.encoder.output_size

    def _build(self, ):
        self.encoder = Patchwork.make(self.encoder, is_nested=True)

        if self.is_variational:
            self.mu = Patchwork.make(self.mu, is_nested=True)
            self.log_var = Patchwork.make(self.log_var, is_nested=True)

            self._log_scale = Parameter(tensor([self.log_scale], device=self.encoder.device), )
            self._log_scale.requires_grad = False

            if self.mu_prime is not None and self.log_var_prime is not None:
                self.mu_prime = Patchwork.make(self.mu_prime, is_nested=True)
                self.log_var_prime = Patchwork.make(self.log_var_prime, is_nested=True)
            else:
                self.mu_prime, self.log_var_prime = None, None

        if self.num_components > 1:
            self.weight = Patchwork.make(self.weight, is_nested=True)

            if self.weight_prime is not None:
                self.weight_prime = Patchwork.make(self.weight_prime, is_nested=True)

        self.decoder = Patchwork.make(self.decoder, is_nested=True)

    @property
    def retain_grad(self):
        return Patchwork.retain_grad.fget(self)

    @retain_grad.setter
    def retain_grad(self, value):
        Patchwork.retain_grad.fset(self, value)
        if self.is_variational:
            self._log_scale.requires_grad = False

    def to(self, device):
        Patchwork.to(self, device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        if self.is_variational:
            self.mu = self.mu.to(device)
            self.log_var = self.log_var.to(device)
            self._log_scale = Parameter(self._log_scale.to(device))

            if self.mu_prime is not None:
                self.mu_prime = self.mu_prime.to(device)

            if self.log_var_prime is not None:
                self.log_var_prime = self.log_var_prime.to(device)

        return self

    def forward(self, x: Tensor, *args: Tuple[Tensor]) -> Tensor:
        # merge flattened features, but keep batch-size and seq-len,
        y = self.encode(x, *args)        
        return self.decode(y)

    def encode(self, x: Tensor, *args: Tuple[Tensor]):
        if not isinstance(x, Tensor):
            x, *args = self.to_tensor(x, *args)

        batch_dim_append = len(x.shape) == 1
        if batch_dim_append:
            x = x.unsqueeze(0)  # add batch dim

        y = self.encoder(x, *args)
        
        if self.is_variational:
            weights, mu, sigma = self.encode_variational_parameters(y)
            y = self.sample_normal(weights, mu, sigma)

        if batch_dim_append:
            return y.squeeze(0)

        return y

    def encode_variational_parameters(self, x_encoding):
        weights = None if self.weight is None else self.weight(x_encoding)
        mu = self.mu(x_encoding)
        sigma = exp(self.log_var(x_encoding) * 0.5)  # CORRESPONDS TO DIAGONAL NOISE IN `MDN`

        if self.weight is not None:
            mu = stack(mu.split(self.num_components, dim=-1), dim=-2)            # (BS, LATENT_SIZE, NUM_COMPONENTS)
            sigma = stack(sigma.split(self.num_components, dim=-1), dim=-2)      # (BS, LATENT_SIZE, NUM_COMPONENTS)
            weights = stack(weights.split(self.num_components, dim=-1), dim=-2)  # (BS, NUM_COMPONENTS, 1)
            weights = Softmax(dim=-1)(weights * 0.5)  # normalize

        return weights, mu, sigma

    def forward_mu(self, x: Tensor, *args: Tuple[Tensor]):
        """ Evaluates the latent space projection of x (in case of a variational auto encoder, the
            result of the `mu` layer is returned without `log_var` sampling). """
        if not isinstance(x, Tensor):
            x, *args = self.to_tensor(x, args)

        batch_dim_append = len(x.shape) == 1
        if batch_dim_append:
            x = x.unsqueeze(0)  # add batch dim

        y = self.encoder(x, *args)

        if self.is_variational:
            weights, mu, _ = self.encode_variational_parameters(y)
            if self.num_components == 1:  # regular VAE
                y = mu
            else:
                y = (mu * weights).sum(dim=-1)

        if batch_dim_append:
            return y.squeeze(0)

        return y

    @staticmethod
    def sample_normal(weights: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """ Evaluates the latent space projection of x (in case of a variational auto encoder, the
            result of the `mu` layer is randomized according to the `log_var` results, providing
            generative samples of the VAE). """
        normal = Normal(mu, sigma)
        if weights is None:
            # define the Normal distribution of encoder
            return normal.rsample()

        else:
            mixture = MixtureSameFamily(
                mixture_distribution=Categorical(weights),
                component_distribution=normal
            )
            return mixture.sample()

    def decode(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x, *_ = self.to_tensor(x)
        batch_dim_append = len(x.shape) == 1
        if batch_dim_append:
            x = x.unsqueeze(0)  # add batch dim

        x = self.decoder(x)

        if self.is_variational and self.mu_prime is not None and self.log_var_prime is not None:
            weights_prime, mu_prime, sigma_prime = self.decode_variational_parameters(x)
            x = self.sample_normal(weights_prime, mu_prime, sigma_prime)

        if batch_dim_append:
            return x.squeeze(0)

        return x

    def decode_variational_parameters(self, z_decoding):
        weights_prime = None if self.weight_prime is None else self.weight_prime(z_decoding)
        mu_prime = self.mu_prime(z_decoding)
        sigma_prime = exp(self.log_var_prime(z_decoding) * 0.5)  # CORRESPONDS TO DIAGONAL NOISE IN `MDN`

        if self.weight_prime is not None:
            mu_prime = stack(mu_prime.split(self.num_components, dim=-1), dim=-2)            # (BS, RECON_SIZE, NUM_COMPONENTS)
            sigma_prime = stack(sigma_prime.split(self.num_components, dim=-1), dim=-2)      # (BS, RECON_SIZE, NUM_COMPONENTS)
            weights_prime = stack(weights_prime.split(self.num_components, dim=-1), dim=-2)  # (BS, RECON_SIZE, NUM_COMPONENTS)
            weights_prime = Softmax(dim=-1)(weights_prime * 0.5)  # normalize

        return weights_prime, mu_prime, sigma_prime

    @staticmethod
    def reconstruction_loss(x_hat, x, r_weights=None) -> Tensor:
        """ Mean Square Error - reconstruction loss

        :param x_hat: reconstruction of x
        :param x: original input (to be reconstructed)
        :param r_weights: Optional item-wise weight for the reconstruction loss, defaults to None.
        """

        if r_weights is not None:
            loss = sum(r_weights * (x_hat - x) ** 2)/x_hat.shape[0]
        else:
            loss = MSELoss()(x, x_hat)
        return loss

    def gaussian_log_likelihood(self, x_hat, x, r_weights=None) -> Tensor:
        """ negative Gaussian Likelihood reconstruction loss: -log(p(x|z))

        measure prob of seeing x_hat under p(x|z)

        :param x_hat: reconstruction of x
        :param x: original input (to be reconstructed)
        :param r_weights: Optional item-wise weight for the reconstruction loss, defaults to None.
                          Doesn't do anything here, see `reconstruction_loss` instead.
        """
        scale = exp(self._log_scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = Normal(x_hat, scale).log_prob(x)
        return -log_pxz.sum(dim=tuple(range(1, len(log_pxz.shape))))  # sum all but batch-dim

    @staticmethod
    def latent_loss(weights, mu, sigma, z) -> Tensor:
        """ Latent-space loss of the generative model, i.e., the KL-divergence of the probability of the samples `z`
            being generated by a Gaussian (mixture density) model to standard normal distribution.

        :param weights: weights of different PDFs if multiple mu and sigma values are provided (last tensor dimension).
        :param mu: mean of encodings
        :param sigma: standard deviation of encodings
        :param z: sampled encodings
        """
        q = Normal(mu, sigma)                         # normal distribution of mu and sigma
        if weights is None:
            log_qzx = q.log_prob(z)

            p = Normal(zeros_like(mu), ones_like(sigma))  # standard normal distribution
            log_pz = p.log_prob(z)
        else:
            q_mixture = MixtureSameFamily(mixture_distribution=Categorical(weights), component_distribution=q)
            log_qzx = -q_mixture.log_prob(z)

            log_pz = -1e-5*(ReLU()((1.-sigma)**2)).mean(-1)/mu.shape[0]
            log_pz -= 1e-5*(mu**2).mean(-1)/mu.shape[0]
            # p = Normal(zeros_like(mu[:, :, 0]), ones_like(sigma[:, :, 0]))  # standard normal distribution
            # log_pz = p.log_prob(z)

        loss = (log_qzx - log_pz)  # KL loss
        loss = loss.sum(-1)
        return loss

    def loss(self, x: Tensor, x_hat: Tensor, weights=None, mu=None, sigma=None, z=None, r_weights=None) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """ total loss function of the AE (or VA, if mu and log_var networks are defined)

        consists of `mean(beta * KL_loss + r_loss)`

        :param x_hat: reconstruction of x
        :param x: original input (to be reconstructed)
        :param weights: weights of different PDFs if multiple mu and sigma values are provided (last tensor dimension).
        :param mu: mean of encodings
        :param sigma: standard deviation of encodings
        :param z: sampled encodings
        :param r_weights: Optional item-wise weight for the reconstruction loss, defaults to None.
        """

        if self.mu_prime is None or self.log_var_prime is None:
            r_loss: Tensor = self.reconstruction_loss(x_hat=x_hat, x=x, r_weights=r_weights)
        else:
            r_loss: Tensor = self.gaussian_log_likelihood(x_hat=x_hat, x=x, r_weights=r_weights)

        if not self.is_variational:  # Regular AutoEncoder
            return r_loss

        # Variational Auto Encoder:
        l_loss: Tensor = self.latent_loss(weights=weights, mu=mu, sigma=sigma, z=z)

        # elbo
        l_loss = l_loss.mean() * self.beta
        r_loss = r_loss.mean()
        loss: Tensor = (l_loss + r_loss).mean()

        return loss, l_loss, r_loss

    def batch_loss(self, batch, batch_idx=None, return_encodings=False, r_weights=None):
        x, *_ = batch

        if not self.is_variational:
            weights, mu, sigma, z = None, None, None, None
            if not return_encodings:
                x_hat = self.forward(x, )

            else:
                z = self.encode(x, )
                x_hat = self.decode(z)

        else:
            # x -> (mu, sigma) -> (q -> z) -> x_hat
            weights, mu, sigma = self.encode_variational_parameters(self.encoder(x, ))
            z = self.sample_normal(weights=weights, mu=mu, sigma=sigma)
            x_hat = self.decode(z)

        loss = self.loss(x=x, x_hat=x_hat, weights=weights, mu=mu, sigma=sigma, z=z, r_weights=r_weights)

        if return_encodings and self.is_variational:
            return loss, (mu, z)

        elif return_encodings:
            return loss, z

        return loss

    # def plot_network(self):
