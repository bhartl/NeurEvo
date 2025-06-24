from mindcraft.agents import TorchAgent
from torch import Tensor, tensor, float32, cat


class SensoryNeuronAgent(TorchAgent):
    """ SensoryNeuronAgent is a TorchAgent that processes observations and actions
    through a foldback mechanism, allowing it to retain and utilize attributes
    from previous steps in the agent's lifecycle.

    Inspired by Tang and Ha [1]

    [1] Y. Tang, D. Ha, The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning, NeurIPS 2021

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ("foldback_attrs", *TorchAgent.REPR_FIELDS)

    def __init__(self,
                 foldback_attrs=(),
                 **kwargs,
                 ):

        # attr helpers
        self.foldback_attrs = foldback_attrs
        self._foldback_buffer = {attr: None for attr in foldback_attrs}

        # init
        TorchAgent.__init__(self, **kwargs)

    def preprocess_observation(self, observation, info):
        observation, *args = TorchAgent.preprocess_observation(self, observation, info)
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(2)
        self.put_foldback_attr(observation=observation)
        observation = self.get_forward_attr()
        return observation

    def get_action(self, observation, reward=None, info=None) -> object:
        action = TorchAgent.get_action(self,  observation=observation, reward=reward, info=info)
        self.put_foldback_attr(reward=reward, info=info, action=action)
        if len(action) == 1:
            action = action[0]
        return action

    def put_foldback_attr(self, **kwargs):
        for attr, v in kwargs.items():
            self._foldback_buffer[attr] = v

    @property
    def default_reward(self):
        return [0.0]

    def get_forward_attr(self):
        x = []
        for k in ["observation", *self.foldback_attrs]:
            v = self._foldback_buffer[k]
            if v is None:
                v = getattr(self, "default_" + k)

            if not isinstance(v, Tensor):
                v = tensor(v, dtype=float32, device=self.device)

            x.append(v)

        return x

    def to_dict(self):
        dict_repr = TorchAgent.to_dict(self)
        dict_repr["foldback_attrs"] = list(dict_repr.get("foldback_attrs", []))
        return dict_repr