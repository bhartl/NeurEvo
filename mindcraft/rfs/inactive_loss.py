from mindcraft import ReferenceFrame
from typing import Optional, Union
import numpy as np
from mindcraft.io import Log


class InactiveLoss(ReferenceFrame):

    DEFAULT_LOSS = ("action", )
    REPR_FIELDS = ('gamma', *ReferenceFrame.REPR_FIELDS)

    def __init__(self,
                 gamma: float = 1.,
                 loss_fields: Optional[Union[list, tuple, str]] = DEFAULT_LOSS,
                 loss_factor: float = 1.0,
                 **patchwork_kwargs
                ):
        """ Constructs a VarianceLoss ReferenceFrame Instance

        :param gamma: Expected minimal variance scale, lower variance values will be penalized by `var_loss`, defaults to 1.
        :param input_fields: List or Tuple of class properties which are the input tensors to the
                             embedding module.
        :param loss_fields: List or Tuple of class properties which are the ground truth of the
                            prediction loss (i.e., which should be predicted correctly by the embedding module)
        :param loss_factor: Float value to weight the `step_loss` and `episode_loss` functions, defaults to 0.
        """

        ReferenceFrame.__init__(self, loss_factor=loss_factor, loss_fields=loss_fields, )
        self.gamma = gamma

    def episode_loss(self, log_history):
        """ Intrinsic loss function for an entire episode which is evaluated on the logged data through an episode

        :param log_history: Log-history of the Agent. If any `loss_fields` other than `observation`, `action` or
                            `reward` are required by the ReferenceFrame, they need to be logged via the `log_fields` and
                            `log_foos` arguments (see `mindcraft.io.Log`).
        :return: ReferenceFrame-loss corresponding to an entire episode, defaults to 0.
        """

        inactive_loss = 0.
        for d_t in [log_history[-1][key] for key in self.loss_fields]:
            inactive_loss -= np.sum(np.abs(d_t[1:] - d_t[:-1]))

        return inactive_loss * self.loss_factor
