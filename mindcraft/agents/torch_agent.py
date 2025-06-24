from mindcraft import Agent
from mindcraft.torch.module import Patchwork
from mindcraft.torch.util import tensor_to_numpy
from mindcraft.io.spaces import space_clip
from torch import tensor, Tensor, float32, cat
from torch.nn import Module
from numpy import ndarray, asarray, atleast_2d, concatenate
from typing import Union, Optional


class TorchAgent(Agent, Module):
    """ An Agent with a single PyTorch `policy_module` with a 'reactive' policy:
        - perceives environment observations without preprocessing
        - proposes actions based on its `policy_module` and internal state (if available)

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('policy_module',
                   'observation_mask',
                   'device',
                   'np_fields',
                   'clip',
                   'retain_grad',
                   'unfreeze',
                   'parameter_scale',
                   *Agent.REPR_FIELDS)

    MODULES = ('policy_module',)

    def __init__(self,
                 policy_module: Union[Patchwork, str, dict],
                 observation_mask: Optional[list] = (),
                 device: str = 'cpu',
                 np_fields: bool = True,
                 clip: bool = True,
                 retain_grad: bool = False,
                 unfreeze: tuple = None,
                 modules: Optional[dict] = None,
                 parameter_scale: float = 1.,
                 **kwargs):
        """ Constructor of `TorchAgent` Class

        :param policy_module: policy module representation (either a `mindcraft.torch.module.MindModule`, `str` or `dict`).
        :param observation_mask: Optional indices or boolean array to mask the observation space
                                 (provided indices or True's are considered, others are masked out).
        :param device: device-specifier for the module and tensors, defaults to "cpu".
        :param np_fields: Boolean controlling whether numpy arrays are returned as action (used for rollouts,
                          defaults to True)
        :param clip:  boolean, specifying whether the action is clipped to the allowed action space values,
                      defaults to True.
        :param retain_grad: Boolean specifying, whether the module is trainable via backpropagation, defaults to False.
                            If set, gradient evaluations are performed and backpropagation might be applicable,
                            however, this might lead to significant memory usage.
        :param unfreeze: List or tuple of models which are trainable, defaults to `MODULES`.
        :param parameter_scale: Scaling factor for the parameters of the agent's models, defaults to 1.
        :param kwargs: Keyword parameters forwarded to `mindcraft.Agent` super-class.
        """
        Module.__init__(self)

        kwargs["to_list"] = list(kwargs.get("to_list", [])) + ["observation_mask", "unfreeze"]
        kwargs["omit_default"] = kwargs.get("omit_default", True)
        Agent.__init__(self, **kwargs)

        self._unfreeze = None
        self.n_parameters = {}
        self.policy_module = Patchwork.make(policy_module)
        # set other torch-modules after `Module.__init__()` and prior to `unfreeze`
        for module_name, module in (modules or {}).items():
            assert module_name != 'policy_module'
            setattr(self, module_name, module)

        self.unfreeze = tuple(unfreeze or self.MODULES)

        self._retain_grad = None
        self.retain_grad = retain_grad
        self.observation_mask = asarray(observation_mask)

        self._device = None
        self.device = device

        # helpers
        self.np_fields = np_fields
        self.dtype = float32
        self.clip = clip
        self.action = None
        self.parameter_scale = parameter_scale
        #  self.reset()

    def __str__(self):
        dict_repr = self.to_dict()
        pop = ['serialized', 'recover_indices', 'serialize_mask']
        [[dict_repr[m].pop(p, None) for p in pop] for m in self.MODULES]
        import yaml
        return yaml.safe_dump(dict_repr, default_flow_style=None)

    @property
    def retain_grad(self):
        return self._retain_grad

    @retain_grad.setter
    def retain_grad(self, value):
        for m in self.get_modules(unfrozen=False):
            assert isinstance(m, Patchwork), f"Sub-Module named '{m.__name__}' not of type `Patchwork`, " \
                                                f"got `{type(m)}` instead."
            m.retain_grad = value

        for c in self.rf:
            try:
                c.retain_grad = value
            except AttributeError:
                pass

        self._retain_grad = value

    @property
    def unfreeze(self):
        return self._unfreeze

    @unfreeze.setter
    def unfreeze(self, value):
        assert len(value) > 0
        self.n_parameters = {}
        for p in value:
            model = self.get_module(p)
            # if model is None:  # could not relate model name `p` with property
            #     try:  # to split identifier '<label>.model.with.weights.or.so' to 'label' and 'model.with.weights.or.so'
            #         critic_label, model_name = p.split('.')[0], '.'.join(p.split('.')[1:])
            #     except IndexError:  # and go with the entire string `p` per default
            #         critic_label, model_name = None, p
            #
            #     for key, critic in self.critic.items():
            #         # then check whether any critic (or a special 'label'ed one) hosts the model
            #         if critic_label is None or key == critic_label:
            #             model = getattr(critic, model_name, None)  # try to get model from critics
            #             if model is not None:  # consider only first appearance of model
            #                 if hasattr(critic, 'serialize_parameters'):
            #                     model = critic
            #                 setattr(self, p, model)
            #                 break

            assert model is not None, f"Could not relate model '{p}' with any property in Agent or its Critics."
            parameters = model.serialize_parameters()
            self.n_parameters[p] = len(parameters) if parameters is not None else -1

        self._unfreeze = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.to(device=value)

    def to(self, device: str):
        """ move module, sub-module and tensors to specified `device` ('cpu' or 'cuda:Number', torch-specific).
            :returns: self
        """
        [m.to(device) for m in self.get_modules(unfrozen=False) if m is not None]  # move models, if not None
        [c.to(device) for c in self.rf.values() if hasattr(c, "to")]           # move critics, if are module

        try:
            self.action = self.action.to(device)
        except AttributeError:
            pass

        self._device = device
        return self

    def reset(self, observation, reward, info):
        for m in self.get_modules(unfrozen=False):
            try:
                m.reset()
            except AttributeError:
                pass

        self.action = tensor(self.default_action, device=self.device, dtype=self.dtype)
        self.to(self.device)
        if not self.np_fields:
            return self.action.clone()

        return tensor_to_numpy(self.action.squeeze(0))

    def preprocess_observation(self, observation, info):
        if not isinstance(observation, Tensor):
            observation = atleast_2d(observation)  # (DIM--OBS) -> (N-BATCH, DIM-OBS)
            observation = tensor(observation, device=self.device, dtype=self.dtype)

        return (self.mask_observation(observation), )

    def mask_observation(self, observation: Tensor):
        if any(self.observation_mask):
            return observation[..., self.observation_mask]

        return observation

    def preprocess_reward(self, reward):
        if reward is None:
            reward = 0.

        if not isinstance(reward, Tensor):
            reward = tensor(reward, device=self.device, dtype=self.dtype)

        if len(reward.shape) == 0:
            reward = reward.unsqueeze(0)  # add batch dimension

        # potentially move to correct device
        return reward.to(self.device)

    def get_recover_indices(self):
        recover_indices = []
        for d in (self.get_module(p).recover_indices for p in self.unfreeze):
            recover_indices.extend([param, [start, end]] for param, (start, end) in d)

        return recover_indices

    def get_module(self, name):
        model = getattr(self, name, None)
        if model is None:  # could not relate model name `p` with property
            try:  # to split identifier '<label>.model.with.weights.or.so' to 'label' and 'model.with.weights.or.so'
                critic_label, model_name = name.split('.')[0], '.'.join(name.split('.')[1:])
            except IndexError:  # and go with the entire string `p` per default
                critic_label, model_name = None, name

            for key, critic in self.critic.items():
                # then check whether any critic (or a special 'label'ed one) hosts the model
                if critic_label is None or key == critic_label:
                    model = getattr(critic, model_name, None)  # try to get model from critics
                    if model is not None:  # consider only first appearance of model
                        setattr(self, name, model)
                        break

        if model is None:
            raise ValueError(f"Module named `{name}' not found in agent '''{str(self)}'''.")

        return model

    def get_modules(self, unfrozen=True):
        return tuple([self.get_module(p) for p in (self.unfreeze if unfrozen else self.MODULES)])

    def get_parameters(self, modules=None):
        """ get all parameters of the agent's `unfreeze`d modules as a single concatenated tensor.

        :param modules: Optional list of modules to get parameters from, defaults to `unfreeze`d modules.
        :returns: either a concatenated tensor of all parameters of the agent's `unfreeze`d modules,
                  or numpy array if `np_fields` is set.
        """
        if modules is None:
            modules = self.unfreeze

        p = [self.get_module(p).serialize_parameters(to_numpy=self.np_fields) for p in modules]
        try:
            assert not any(pi is None for pi in p)
            if self.np_fields:
                p = concatenate(p)

            else:
                p = cat(p)

            return p / self.parameter_scale

        except (AssertionError, ValueError):
            return None

    def set_parameters(self, parameters, modules=None):
        if parameters is None:
            return

        if modules is None:
            modules = self.unfreeze

        n_prev = 0
        parameters = parameters * self.parameter_scale
        for trainable in modules:
            model = self.get_module(trainable)
            n_model = self.n_parameters[trainable]
            if n_model >= 0:
                n_model += n_prev
                model_parameters = parameters[n_prev:n_prev + n_model]
                model.deserialize_parameters(model_parameters)
                n_prev = n_model

            else:  # assume NEAT, parameters -> (genome, config)
                assert len(modules) == 1
                model.deserialize_parameters(parameters)

    def forward(self, observation, reward=None, info=None) -> Union[ndarray, Tensor, tuple, list]:
        """ evaluate the agent's action from an `observation` of the environment by applying the agent's

        - A policy-module controller proposes actions from current observations and possible hidden-state
        - and clips the result to the agent's action space boundaries.

        :param observation: environmental observation, numpy array or pytorch tensor.
        :param reward: Optional reward signal from the environment.
        :param info: Optional info dictionary signal from the environment.
        :returns: proposed action, clipped to agent's action space boundaries (if `clip` is set),
                  as `numpy.ndarray` if the `np_fields` is set, as `torch.Tensor` otherwise.
                  If only one single observation is provided (i.e., if the batch-size is 1), the
                  return value is flattened to the action-space shape.
        """

        if observation is None:
            return self.default_action

        observation, *args = self.preprocess_observation(observation, info)
        squeeze = len(observation.shape) == 2
        if self.policy_module.is_sequence_module and squeeze:
            observation = observation.unsqueeze(1)

        action = self.policy_module(observation, *args)
        action = self.forward_action(action, squeeze)
        return action

    def forward_action(self, action, squeeze=True):
        if self.clip:
            # clip values to action_space, clipping with array-like min and max only in numpy
            action = space_clip(action, self.action_space)

        self.action = action
        if self.np_fields:
            # transform to numpy
            action = tensor_to_numpy(action)

        if action.shape[0] == 1 and len(action.shape) > 1 and squeeze:
            # remove batch-dim
            return action[0]

        return action

    @property
    def hidden_state(self):
        try:
            hidden_state = self.policy_module.hidden_state
            return hidden_state

        except AttributeError:
            return None

    @property
    def observation_size(self):
        return self.policy_module.input_size

    @property
    def action_size(self):
        return self.policy_module.output_size
