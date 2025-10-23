import argh
from pydoc import locate
import numpy as np
try:
    from gymnasium import Env as GymEnv
except ImportError:
    from gym import Env as GymEnv
from mindcraft.io import Repr
from mindcraft.io.spaces import Space
from typing import Tuple, Union


class Env(GymEnv, Repr):
    """ Base-class for every `mindcraft`-environment located in `mindcraft.envs` which inherits from `gym.Env`.

    - Every agent-environment pair comes with an `action_space` and an `observation_space`.
      These attributes are of type Space, and they describe the format of valid actions and observations

    For more details see https://gym.openai.com/docs/

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('render_on_reset', )  # , 'observation_space')

    DEFAULT_LOCATE = 'mindcraft.envs'

    def __init__(self,
                 observation_space: Union[Space, str] = None,
                 reward: float = 0.,
                 steps: int = 0,
                 verbose: bool = False,
                 render_on_reset: bool = True,
                 **kwargs
                 ):

        Repr.__init__(self, repr_fields=self.REPR_FIELDS, omit_default=True)
        GymEnv.__init__(self)

        self._observation_space = None
        self.observation_space = observation_space

        self.reward = reward
        self.steps = steps
        self.verbose = verbose
        self.render_on_reset = render_on_reset

    @property
    def observation_space(self) -> Union[Space, None]:
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Union[Space, str, None]):
        if isinstance(value, str):
            value = eval(value)

        assert isinstance(value, Space) or value is None, f"observation_space type {type(value)} not supported"
        self._observation_space = value

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """ The implementation of the classic “agent-environment loop”.

        Each time-step, the agent chooses an action, and the environment returns an observation and a reward.

        The environment’s step function returns four values:

        - observation (object): an environment-specific object representing your observation of the environment.
        - reward (float): amount of reward achieved by the previous action.
        - done (boolean): whether it’s time to reset the environment again.
        - info (dict): diagnostic information useful for debugging.
        """
        return None, self.reward, True, {}

    def reset(self, action: object = None) -> object:
        """
        :param action: (object) default action for initial step (defaults to None)
        :returns: tuple of (initial observation, reward, done, info)
        """
        self.reward = 0.
        self.steps = 0

        observation, *_ = self.step(action)
        return observation, 0., False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class WorldmodelEnv(Env):
    """ A wrapper class for a worldmodels-environment.

    (c) B. Hartl 2021
    """
    def __init__(self, agent, *args, **kwargs):
        Env.__init__(self, *args, **kwargs)

        self.agent = agent

        self.encoding = None
        self.hidden_state = None
        self.reward = None

        self.cv2 = None
        self.reset()

    def reset(self, action: object = None):
        self.encoding = self.get_default_encoding()
        self.hidden_state = self.get_default_hidden_state()
        self.reward = self.get_default_reward()
        return self.encoding

    def get_default_encoding(self):
        return None

    def get_default_hidden_state(self):
        return None

    def get_default_reward(self):
        return None

    def encode(self, observation) -> object:
        """ Encode the Agent's latent (dream) state by encoding an observation """
        encoding = self.request_agent_encoding(observation)
        self.set_encoding(encoding)
        return encoding

    def request_agent_encoding(self, observation):
        """ Encode the Agent's latent (dream) state by encoding an observation """
        return self.agent.encode(observation)

    def set_encoding(self, value=None):
        """ Set the Agent's latent (dream) state """
        self.encoding = value

    @property
    def dream_reconstruction(self):
        return self.agent.decode(self.encoding)

    def step(self, action):
        # apply mdn-rnn model
        hidden_state, next_encoding, next_reward = self.agent.predict(self.encoding, action, self.reward)
        observation, reward = self.update_memory(hidden_state, next_encoding, next_reward)
        return observation, reward, False, {}

    def update_memory(self, hidden_state, encoding, reward):
        self.hidden_state = hidden_state
        self.set_encoding(encoding)
        self.reward = reward

        return self.encoding, self.reward

    def render(self, mode='human'):
        if self.cv2 is None:
            import cv2
            self.cv2 = cv2

        obs = self.dream_reconstruction
        if isinstance(obs, np.ndarray):
            if obs.dtype in (np.float, np.float32):
                obs = (obs * 255).astype('uint8')

        obs = self.cv2.resize(obs, (256, 256))
        self.cv2.imshow('indream', obs)
        self.cv2.waitKey(1)

    def close(self):
        if self.cv2 is not None:
            self.cv2.destroyAllWindows()


def demo(env='car_racing'):
    """ apply demo environments, implemented in the `mindcraft.envs` module

    Note: these `envs` must implement a callable `demo` function

    :param env: env callable or python module (e.g. python file without '.py' ending) located in  `mindcraft.envs`
                (defaults to 'car_racing').
    """
    if isinstance(env, str):
        located_env = locate(env)
        if located_env is None:
            located_env = locate('mindcraft.envs.' + env)
        assert located_env is not None, f'Did not understand env `{env}`'
        env = located_env

    return env.demo()


def gym_registry():
    """ Prints all rigestered gym environments to the screen. """
    from gym import envs
    print(envs.registry.all())


if __name__ == '__main__':
    argh.dispatch_commands([
        demo,
        gym_registry,
    ])

