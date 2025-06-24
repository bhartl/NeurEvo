from mindcraft import ReferenceFrame
from mindcraft.torch.module import Patchwork
from torch import Tensor, concatenate, mean, sum
from torch.nn import Module
from typing import Optional, Union


class EmbeddingPrediction(ReferenceFrame, Patchwork):

    DEFAULT_EMBED = ("embedding", )
    REPR_FIELDS = ('embedding_module', 'input_fields',
                   *ReferenceFrame.REPR_FIELDS, *Patchwork.REPR_FIELDS)

    def __init__(self,
                 nn: Union[Patchwork, dict, str],
                 input_fields: Optional[Union[list, tuple, str]],
                 loss_fields: Optional[Union[list, tuple, str]] = DEFAULT_EMBED,
                 loss_factor: float = 1.0,
                 **patchwork_kwargs
                ):
        """ Constructs a ReferenceFrame Instance

        :param nn: A `mindcraft.torch.module.Patchwork` representable that outputs predictions
                   (based on input tensors labeled via `input_fields`) for embeddings (labeled
                   via `loss_fields`).
        :param input_fields: List or Tuple of class properties which are the input tensors to the
                             embedding module.
        :param loss_fields: List or Tuple of class properties which are the ground truth of the
                            prediction loss (i.e., which should be predicted correctly by the embedding module)
        :param loss_factor: Float value to weight the `step_loss` and `episode_loss` functions, defaults to 0.
        """

        ReferenceFrame.__init__(self, loss_factor=loss_factor, loss_fields=loss_fields, )
        self.input_fields = input_fields

        Module.__init__(self)
        self.nn = nn
        Patchwork.__init__(self, **patchwork_kwargs)
        self.prediction = None

    def _build(self):
        self.nn = Patchwork.make(self.nn)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        to_list = ['loss_fields', 'input_fields']
        for k in to_list:
            if k in dict_repr:
                dict_repr[k] = list(dict_repr[k]) if dict_repr[k] is not None else None
        return dict_repr

    def to(self, device: str):
        """ move embedding module to specified `device` (cpu or cuda)
        :param device: Name of the specified `device`, e.g., 'cpu' or 'cuda'.
        :returns: self
        """
        self.embedding_module = self.embedding.to(device)
        return self

    def step_loss(self, observation, action, reward, log_history):
        """ Intrinsic loss function, penalizing the prediction errors of concatenated `loss_fields` with the
            predictions from the embedding module.

        :param log_history: Log-history of the Agent. If any `loss_fields` other than `observation`, `action` or
                            `reward` are used by the ReferenceFrame, they need to be logged via the `log_fields` and
                            `log_foos` arguments (see `mindcraft.io.Log` for details).
        :return: Weighted (by `loss_factor`) MSE loss of the prediction error.
        """
        x = self.get_embedding_input(observation, action, reward, log_history)
        prediction = self.forward(*x)
        embedding = self.get_embedding_target(observation, action, reward, log_history)
        prediction_error = self.get_prediction_error(embedding, prediction)
        return prediction_error * self.loss_factor

    def get_embedding_input(self, observation, action, reward, log_history=None, step=-1, **kwargs) -> tuple:
        x = []
        log_history = log_history or self.log_history
        for key in self.input_fields:
            if key in locals():
                x.append(locals()[key])
            elif hasattr(self, key):
                x.append(getattr(self, key))
            elif key in kwargs:
                x.append(kwargs[key])
            else:
                x.append(self.get_step_data_by_key(key=key, log_history=log_history, step=step))

        return tuple(x)

    def get_embedding_target(self, observation, action, reward, log_history=None, step=-1, **kwargs):
        y = []
        log_history = log_history or self.log_history
        for key in self.loss_fields:
            if key in locals():
                y.append(locals()[key])
            elif hasattr(self, key):
                y.append(getattr(self, key))
            elif key in kwargs:
                y.append(kwargs[key])
            else:
                y.append(self.get_step_data_by_key(key=key, log_history=log_history, step=step))

        return concatenate(y)

    def forward(self, x, *args):
        x = Patchwork.forward(x, *args)
        y = self.nn(x)
        self.prediction = y
        return y

    @staticmethod
    def get_prediction_error(embedding, prediction) -> Union[float, Tensor]:
        """ Evaluates the MSE prediction error for sequentially attributes
        :return: The square error of the embedding applied to the `loss_fields` attribute specified with `key`.
        """
        loss = mean(sum((embedding - prediction)**2, dim=-1), dim=0)
        return loss

    def reset(self):
        self.embedding_module.reset()
        self.prediction = None

    @property
    def hidden_state(self):
        return self.embedding_module.hidden_state
