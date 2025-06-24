import yaml
from typing import Union, Optional
from mindcraft.io import Repr


class ReferenceFrame(Repr):
    """ A wrapper class for `mindcraft.Agent` Instances to implement a reference frame to evaluate
        an intrinsic loss for single steps (`step_loss`) or for entire episodes (`episode_loss`).

    An `Agent` can have associated a single or several `ReferenceFrame` instance, which return their `step_loss` or
    `episode_loss` evaluations based on the `Agent`'s `log_history`.

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('loss_factor',
                   'loss_fields',
                   )

    DEFAULT_LOCATE = 'mindcraft.rfs'

    def __init__(self,
                 loss_factor: float = 0.,
                 loss_fields: Optional[Union[list, tuple, str]] = (),
                 ):
        """ Constructs a ReferenceFrame Instance

        :param loss_factor: Float value to weight the `step_loss` and `episode_loss` functions, defaults to 0.
        :param loss_fields: (Optional) List or Tuple of class properties which are subjected to
                            mutual information loss (i.e., temporally successive realizations of
                            these attributes of similar value are favourably, while dissimilar values
                            are unfavourably counted by a cosine-similarity loss term), defaults to `()`.
                            *Note*: attributes need to be logged via `log_fields` by the Agent.
        """
        Repr.__init__(self, repr_fields=self.REPR_FIELDS, omit_default=True)
        self.loss_factor = loss_factorworldmod

        self._loss_fields = None
        self.loss_fields = loss_fields

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_dict(), default_flow_style=None)

    @property
    def loss_fields(self):
        """ Tuple of class-attribute names that are used for the `step_loss` or `episode_loss` functions """
        return self._loss_fields

    @loss_fields.setter
    def loss_fields(self, value: Union[tuple, list, str]):
        """ Tuple of class-attribute names that are used for the `step_loss` or `episode_loss` functions

        :param value: List, Tuple or single str, defining the class attributes related to the `loss_fields`.
        """
        if not hasattr(value, '__iter__') or isinstance(value, str):
            value = (value, )

        self._loss_fields = tuple(value)

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        to_list = ['loss_fields', ]
        for k in to_list:
            if k in dict_repr:
                dict_repr[k] = list(dict_repr[k]) if dict_repr[k] is not None else None
        return dict_repr

    @staticmethod
    def get_step_data(key: str, log_history=None, step=-1, **kwargs):
        """ Helper function to retrieve the parameter list arguments
            (observation, action, reward, legged attributes within log_history)
            via the `key` specifier
        """
        if key in kwargs:
            return kwargs[key]

        assert log_history is not None, f"Key {key} neither in `kwargs` ({list(kwargs.keys())}) nor in `log_history`."
        h5_time_data = log_history[-1][key]
        h5_keys = sorted(h5_time_data.keys())
        if isinstance(step, int):
            return h5_time_data[h5_keys[step]]
        return [h5_time_data[k] for k in h5_keys[step]]

    def reset(self):
        """ Reset internal properties of the RFrame. """
        pass

    def step_loss(self, observation, action, reward, log_history):
        """ Intrinsic loss function which the agent can acquire during execution

        :param observation: Current agent observation.
        :param action: Action the Agent currently proposed.
        :param reward: Previous agent reward.
        :param log_history: Log-history of the Agent. If any `loss_fields` other than `observation`, `action` or
                            `reward` are required by the RFrame, they need to be logged via the `log_fields` and
                            `log_foos` arguments (see `mindcraft.io.Log`).
        :return: Instantaneous ReferenceFrame-loss, defaults to 0.
        """
        return 0.

    def episode_loss(self, log_history):
        """ Intrinsic loss function for an entire episode which is evaluated on the logged data through an episode

        :param log_history: Log-history of the Agent. If any `loss_fields` other than `observation`, `action` or
                            `reward` are required by the ReferenceFrame, they need to be logged via the `log_fields` and
                            `log_foos` arguments (see `mindcraft.io.Log`).
        :return: ReferenceFrame-loss corresponding to an entire episode, defaults to 0.
        """
        return 0.

    def get_parameters(self):
        """ Gets optimization parameters of the ReferenceFrame, if there are any """
        pass

    def set_parameters(self, parameters):
        """ Sets optimization parameters of the ReferenceFrame, if there are any

        :param parameters: Set of parameters of the `ReferenceFrame`.
        """
        pass
