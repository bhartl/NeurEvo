from torch.nn import Module, ReLU
from torch import Tensor, cov, sqrt, diagonal, mean, abs
from mindcraft.torch.module import Patchwork


class VaCoEmbedding(Patchwork):
    """ `Mindcraft` wrapping of Pytorch module implementing a Variance-Covariance embedding based on [1],
      that maintains an `expand` network, whose `embedding` is used to
    - encourage finite variance values in each embedding dimension to maximize information content, and to
    - encourage de-correlated embedding dimensions by minimizing the cross-correlation between different dimensions.


    [1] VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning,
        by A. Bardes, J. Ponce, Y. LeCun, `Arxiv:2105.04906v3 <https://arxiv.org/pdf/2105.04906.pdf>`__
    """

    REPR_FIELDS = ("expand", "gamma", "mean_loss", "var_loss", "cov_loss", "epsilon", *Patchwork.REPR_FIELDS)

    def __init__(self, expand: Patchwork, gamma=1., mean_loss=0.01, var_loss=1.0, cov_loss=0.1, epsilon=1e-4, **patchwork_kwargs):
        """ Constructs a VaCoEmbedding instance.

        :param expand: Patchwork module performing the expansion from the input(latent) space towards the embedding space.
        :param gamma: Expected variance scale, lower variance values will be penalized by `var_loss`, defaults to 1.
        :param mean_loss: Loss factor for constraining the mean of each latent distribution to be located around 0, defaults to 0.01.
        :param var_loss: Loss factor for the variance loss, defaults to 1.
        :param cov_loss: Loss factor for the covariance loss, defaults to 0.1.
        :param epsilon: Numerical stability factor.
        :param patchwork_kwargs:
        """
        Module.__init__(self)
        self.expand = expand
        self.gamma = gamma
        self.mean_loss = mean_loss
        self.var_loss = var_loss
        self.cov_loss = cov_loss
        self.epsilon = epsilon
        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self):
        self.expand = Patchwork.make(self.expand)
        self.expand.is_nested = True

    def to(self, device: str):
        """ move expand model to specified `device` (cpu or cuda)
        :param device: Name of the specified `device`, e.g., 'cpu' or 'cuda'.
        :returns: self
        """
        self.expand = self.expand.to(device)
        return self

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        return dict_repr

    def forward(self, x, embedding=False):
        """ Forward `x` if `not embedding`, otherwise use `expand` network to return `embedding` of `x` """
        return x if not embedding else self.expand(x)

    def get_variance_covariance_loss(self, *embeddings):
        """ Evaluates the variance- and covariance loss terms (Eq. (1) and (4) in Ref. [1])

        :return: The square error of the embedding applied to the `loss_fields` attribute specified with `key`.
        """

        var_loss = 0.
        cov_loss = 0.
        mean_loss = 0.
        for embedding in embeddings:
            if embedding.shape[0] > 1:
                # mean_loss = abs(embedding.mean(dim=0)).sum()
                mean_loss = abs(embedding**2).sum()

                # evaluate covariance matrix of data
                cov_matrix = cov(embedding.T)  # Eq. (3) of Ref. [1]

                # evaluate the variance regularization term, v, of Eq. (1) in Ref. [1]
                std_deviation = sqrt(diagonal(cov_matrix) + self.epsilon)  # Eq. (2) in Ref. [1]: sqrt(var + eps)
                var_loss += mean(ReLU()(self.gamma - std_deviation))       # Eq. (1) in Ref. [1]

                # evaluate the covariance regularization term, c, of Eq. (4) in Ref. [1]
                cov_loss += mean(cov_matrix.fill_diagonal_(0.)**2)         # Eq. (4) in Ref. [1]

        return mean_loss, var_loss, cov_loss

    def loss(self, x: Tensor):
        """ loss function of the VaCoExpand module

        consists of `gamma x variance + delta x covariance` loss

        :param x: original input (to be reconstructed)
        """

        y = self.forward(x, embedding=True)
        mean_loss, var_loss, cov_loss = self.get_variance_covariance_loss(y)
        return mean_loss * self.mean_loss, var_loss * self.var_loss, cov_loss * self.cov_loss


if __name__ == '__main__':
    from mindcraft.torch.module import Recurrent, FeedForward
    from torch import randn
    rgrn = Recurrent(input_size=10, hidden_size=4, output_size=5,
                     layer_type="RGRN", layer_kwargs=dict(inner_merge=True, regulate_x=True, num_iterations=1,))
    expand = FeedForward(input_size=4, hidden_size=[8, 16], output_size=32, activation=["GELU", "GELU", None])
    vaco = VaCoEmbedding(expand=expand)

    # Test the complex-valued layer
    x = randn(64, 10, 10) * 10.
    y = rgrn(x)
    state = rgrn.states[0].squeeze(0)
    loss = vaco.loss(state)
    print(loss)

