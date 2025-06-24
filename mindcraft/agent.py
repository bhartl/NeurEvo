from mindcraft.io.spaces import *  # required to load Box, Discrete, Spaces, etc.
from mindcraft.io import Repr
from mindcraft.io import Log
from mindcraft import ReferenceFrame
from inspect import getmembers, isclass
import numpy as np
import scipy as sp
from collections.abc import Iterable
from collections import OrderedDict
from typing import Union, Optional, List, Tuple
from torch import Tensor


class Agent(Log, Repr):
    """ Base-class for `mindcraft.agents` which can propose actions and an intrinsic loss estimate based on
      `mindcraft.Env` observations in a `mindcraft.World` composition.

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('action_space',
                   'default_action',
                   'reinforcement',
                   'rf',
                   'log_fields',
                   'log_foos',
                   'sampling',
                   'sampling_scale',
                   'sampling_reset',
                   )

    DEFAULT_LOCATE = 'mindcraft.agents'

    POSITIVE_REINFORCEMENT = 'positive'
    NEUTRAL_REINFORCEMENT = 'neutral'
    DEFAULT_REINFORCEMENT = NEUTRAL_REINFORCEMENT

    def __init__(self,
                 action_space: Space,
                 default_action: Union[np.ndarray, Iterable, int, float] = None,
                 reinforcement: str = DEFAULT_REINFORCEMENT,
                 rf: Optional[Union[ReferenceFrame, dict]] = None,
                 log_fields: Union[tuple, list] = Log.DEFAULT_LOG_FIELDS,
                 log_foos: Optional[dict] = None,
                 sampling: Optional[int] = None,
                 sampling_scale: Optional[int] = None,
                 sampling_reset: bool = False,
                 **kwargs
                 ):
        """ Constructs an `Agent` instance

        :param action_space: A representation of the Agent's ActionSpace, can be str or Space instance.
        :param default_action: Iterable, int or float, representing the Agent's default action.
        :param reinforcement: Parameter controlling the merge_rewards_signal weighting:
                                - if set to `POSITIVE_REINFORCEMENT`, only positive `reward - loss`-signals are selected,
                                - otherwise, everything is cumulatively summed for the total reward (defaults to False).
        :param rf: Dictionary of mindcraft.RefFrame makeables, to evaluate step-loss or episode loss on
                       successive or global set of (log) data.
        :param log_fields: tuple or list of strings, specifying variables
                           in the `decide`-method (i.e., 'observation', 'action', 'reward'),
                           in the `step_loss`-method (i.e., 'step_loss', 'step_rating'),
                           in the `episode_loss`-method (i.e., 'episode_loss', 'episode_rating'),
                           or keys in the `log_foos` dictionary,
                           which should be logged in a dictionary for each rollout.
                           One dict with lists for each log-field is generated per rollout.
                           The logs can be retrieved via the Agent's `log_history` property
        :param log_foos: key - value pairs which specify additional log-functions, where the values are either
                         `callable`s or string-code-snippets which can be subjected to the `compile` function.
                         If a log-field (in the `log` argument) corresponds to a key in the `log_foo` dict, the
                         value is executed (if it is callable) or subjected to `eval` otherwise.
                         The result is stored in the world's `log` property.
                         Such a log-foo key value pair could be `steps="self.env.steps"``, to log the step counts
                         of the environment.
        :param sampling: Optional integer specifying the interval after which an action is updated, i.e., how
                         many samples of an observation are considered before an action update is performed,
                         defaults to None. In case of `sampling_scale==-1`, the `sampling` parameter is interpreted
                         as a probability threshold, and the action is updated with probability `sampling`.
        :param sampling_scale: Optional integer specifying the scale of a cdf (cumulative Gaussian distribution function)
                               which will deside when the action-sampling will be executed.
                               If `sampling_scale == 0` (as per default), the action sampling will be done each
                               `sampling` steps.
                               Otherwise, if `sampling_scale > 0`, the sampling will be done at random,
                               following a cdf with mean `sampling` and scale `sampling_scale` according to
                               `rand() < cdf(step, loc=sampling, scale=sampling_scale)`.
                               If `sampling_scale == -1`, the `sampling` parameter is interpreted as a probability.
        :param sampling_reset: Flag to enable initial sampling disregarding other `sampling` settings.
                               Defaults to False. A reset is triggered by the `info` dictionary of the environment.

        """

        kwargs["omit_default"] = kwargs.get("omit_default", True)
        Repr.__init__(self, repr_fields=self.REPR_FIELDS, **kwargs)
        Log.__init__(self, log_fields=log_fields, log_foos=log_foos)
        self.action_space = action_space
        if hasattr(default_action, '__iter__'):
            self.default_action = np.asarray(default_action)
        else:
            self.default_action = np.full(shape=self.action_space.shape,
                                          fill_value=default_action,
                                          dtype=self.action_space.dtype)
        self.sampling = sampling
        self.sampling_scale = sampling_scale
        self.sampling_reset = sampling_reset
        self._sampling_count = 0

        self._reinforcement = None
        self.reinforcement = reinforcement

        self._rf = None
        self.rf = rf
        self.total_reward = 0.
        self.total_loss = 0.
        self.total_rating = 0.

    @classmethod
    def make(cls, repr_obj, **partial_local):
        try:
            return super(Agent, cls).make(repr_obj, **partial_local)

        except NameError:  # add gym spaces to locals() of cls.make ...
            from mindcraft.io import spaces
            partial_make = {name: class_ for name, class_ in getmembers(spaces, isclass)}
            return super(Agent, cls).make(repr_obj, **{**partial_make, **partial_local})

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        dict_repr['action_space'] = repr(dict_repr['action_space'])
        for k in dict_repr.keys():
            if isinstance(dict_repr[k], np.ndarray):
                dict_repr[k] = dict_repr[k].tolist()
        dict_repr['rf'] = {k: c.to_dict() for k, c in self.rf.items()}
        dict_repr['log_fields'] = list(self.log_fields)
        return {k: v for k, v in dict_repr.items() if v is not None and v not in ((), {})}

    def log(self, **kwargs):
        kwargs = self._flatten_rf(**kwargs)
        return Log.log(self, **kwargs)

    def _flatten_rf(self, **kwargs) -> dict:
        """ Flattens all class properties associated with any rf `loss_fields`"""
        attrs = {}
        keys = sum([list(c.loss_fields) for c in self.rf.values() if c is not None], [])
        for k in keys:
            try:
                attr = kwargs[k]
            except KeyError:
                continue

            try:
                attrs[k] = attr.flatten()
            except AttributeError:
                attrs[k] = attr

        return {**attrs, **{k: v for k, v in kwargs.items() if k not in attrs}}

    def reset(self, observation, reward, info):
        """ Abstract method to reset agent, e.g. to remove internal state, etc. """
        return self.reset_base()

    def reset_base(self):
        """ Abstract method to reset agent, e.g. to remove internal state, etc. """
        self.reset_rf()
        self.reset_loss()
        self.reset_sampling()
        return None

    def reset_rf(self):
        """ Abstract method to reset agent, e.g. to remove internal state, etc. """
        # reset reference frames
        return [c.reset() for c in self.rf.values() if c is not None]

    def reset_loss(self):
        """ Abstract method to reset agent, e.g. to remove internal state, etc. """
        # reset loss signals
        self.total_reward = 0.
        self.total_loss = 0.
        self.total_rating = 0.
        return

    def reset_sampling(self):
        """ Abstract method to reset agent, e.g. to remove internal state, etc. """
        self._sampling_count = 0
        return self._sampling_count

    @property
    def rf(self) -> OrderedDict:
        """ Dictionary of `mindcraft.ReferenceFrame`s """
        return self._rf

    @rf.setter
    def rf(self, value):
        """ Dictionary of `mindcraft.ReferenceFrame`s """
        if value is None:
            self._rf = {}
            return

        if not isinstance(value, dict):
            if isinstance(value, (list, dict)):
                value = {i: v for i, v in enumerate(value)}
            else:
                value = {'rf': value}

        self._rf = OrderedDict([(k, ReferenceFrame.make(v)) for k, v in value.items()])

    @property
    def n_rfs(self) -> int:
        """ The number of associated `Critic`s"""
        return len(self.rf)

    @property
    def loss_factors(self) -> tuple:
        """ Retrieve the `loss_factor`s of all `Critic`s """
        return tuple([c.loss_factor for c in self.rf.values()])

    @loss_factors.setter
    def loss_factors(self, value: Union[None, List[Union[float, None]], Tuple[Union[float, None], dict]]):
        """ Update the `loss_factor`s of all `Critic`s

        :param value: Float, or tuple of floats and Nones, to update the corresponding `Critic`'s `loss_factor`s.
                      If a single float is provided, all `Critic`s' `loss_factor''s are updated with that value.
                      If a tuple/list is provided, only floats are considered for updating values,
                      while Nones are used as placeholders to retain the corresponding `Critic`'s `loss_factor`s.
        """
        if value in (None, (), {}, []):
            return

        if not hasattr(value, '__iter__'):
            value = [value] * self.n_rfs

        if not isinstance(value, dict):
            assert len(value) == len(self.rf)
            value = {k: v for k, v in zip(self.rf.keys(), value)}

        for k, v in value.items():
            if v is not None:
                self.rf[k].loss_factor = v

    @property
    def action_space(self) -> (Space, None):
        return self._action_space

    @action_space.setter
    def action_space(self, value: (Space, str, None)):
        if isinstance(value, str):
            from numpy import float64, float32, float16
            from numpy import int64, int32, int16, int8
            from numpy import uint64, uint32, uint16, uint8
            value = eval(value)

        assert isinstance(value, Space) or value is None, f"action_space type {type(value)} not supported"
        self._action_space = value

    @property
    def reinforcement(self):
        return self._reinforcement

    @reinforcement.setter
    def reinforcement(self, value):
        assert value in (self.POSITIVE_REINFORCEMENT, self.NEUTRAL_REINFORCEMENT)
        self._reinforcement = value

    @property
    def is_positive_reinforced(self):
        return self.reinforcement == self.POSITIVE_REINFORCEMENT

    @property
    def is_neutral_reinforced(self):
        return self.reinforcement == self.NEUTRAL_REINFORCEMENT

    def get_action(self, observation, reward=None, info=None) -> object:
        """ The central method of the agent:
            triggers the agent to predict an action (to make a choice)
            based on the observation (and an optional reward) signal,
            and potentially on the Agent's internal states.

        The Method allows the AGent to operate either in the real environment, or in a dream state.

        *Note*: All, observation, reward or the action signals can be logged.

        :param observation: An observation of the environment or imagined state.
        :param reward: An optional reward signal of a prior action, either from the environment or imagined.
        :param info: Optional info dictionary.
        :return: An action, proposing the Agent's next move in the environment.
        """
        action = self.get_default_action()
        try:
            action = self.forward(observation, reward, info)
            if info and info.get("reset", False) and not self.sampling_reset:
                return action
            return self.sample_action(action)

        finally:
            self.log(action=action, observation=observation, reward=reward)

    def sample_action(self, action: Union[np.ndarray, Tensor, None]):
        """ Method to sample a proposed action each `self.sampling` time steps, and forwards a default action instead.

        :param action: The proposed action by the internal processes of the agent.
        """
        if not self.sampling and self.sampling_scale != -1:
            return action

        if action is None:
            action = self.default_action

        is_multiagent = len(action.shape) > 1
        if not self.sampling_scale:  # 0 or None
            forward_action = not self._sampling_count % np.rint(self.sampling)

        else:
            if self.sampling_scale == -1:
                threshold = self.sampling
            else:
                threshold = sp.stats.norm.cdf(self._sampling_count, loc=self.sampling, scale=self.sampling_scale)

            if not is_multiagent:
                forward_action = np.random.rand() <= threshold
            else:  # multi-agent
                forward_action = (np.random.rand(len(action)) <= threshold) & bool(threshold)
                if not hasattr(self._sampling_count, "__iter__"):
                    self._sampling_count = np.full((len(forward_action),), fill_value=self._sampling_count)

        if isinstance(forward_action, bool) and not forward_action:
            action[..., :] = self.default_action
            self._sampling_count = 0

        elif is_multiagent:
            action[~forward_action] = self.default_action[None, ...]
            self._sampling_count[forward_action] = 0

        self._sampling_count += 1
        return action

    def get_step_loss(self, observation, reward, action):
        """ Evaluates the `step_loss` functions all `Critic`s of the Agent

        :param observation:
        :param action:
        :param reward:
        :return:

        Note: Each Critic's `step_loss` is only evaluated if it has a finite `loss_factor`.
        """
        loss = [(c.loss_factor, c.step_loss(observation, action, reward, self.log_history))
                # if c.loss_factor else (0., 0.)
                for c in self.rf.values()]
        weight, loss = [v[0] for v in loss], [v[1] for v in loss]
        rating, loss = self.merge_loss_signals(reward=reward, loss=loss, weight=weight)

        self.total_reward += reward
        self.total_loss += np.sum(loss)
        self.total_rating += rating

        return rating, loss

    def get_episode_loss(self):
        """

        :return:

        Note: Each Critic's `episode_loss` is only evaluated if it has a finite `loss_factor`.
        """
        try:
            loss = [(c.loss_factor, c.episode_loss(self.log_history))
                    # if c.loss_factor else (0., 0.)
                    for c in self.rf.values()]
            weight, loss = [v[0] for v in loss], [v[1] for v in loss]
            episode_rating, episode_loss = self.merge_loss_signals(reward=self.total_rating, loss=loss, weight=weight)
            return episode_rating, episode_loss

        finally:
            self.log_episode += 1

    def merge_loss_signals(self, reward, loss, weight):
        weighted_loss = np.sum([c_loss * c_weight for c_weight, c_loss in zip(weight, loss)])
        rf = reward - weighted_loss
        if self.is_positive_reinforced and rf < 0.:
            return 0., 0.

        return rf, weighted_loss

    def get_default_action(self) -> object:
        return self.default_action

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError()

    def forward(self, observation, reward=None, info=None) -> Union[np.ndarray, Tensor, tuple, list]:
        raise NotImplementedError()

    @property
    def observation_size(self):
        raise NotImplementedError

    @property
    def action_size(self):
        raise NotImplementedError
