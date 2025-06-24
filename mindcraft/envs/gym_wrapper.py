import gym
try:
    import pybullet_envs
except AttributeError:  # XServer issues
    print("#############################################################################################")
    print("### KNOWN BUG IN pybullet_envs/__init__.py:                                               ###")
    print("### FOLLOWING 'https://github.com/openai/gym/issues/3097', SIMPLY                         ###")
    print("### REPLACE `if id in registry.env_specs` WITH `if id not in gym.envs.registry` IN LINE 6 ###")
    print("#############################################################################################")
    raise
except ModuleNotFoundError:
    # pybullet_envs not installed
    pybullet_envs = None
try:
    from stable_baselines3.common.env_util import make_vec_env
except AttributeError:  # cv2 issue
    make_vec_env = None

import numpy as np
from scipy.special import softmax

from mindcraft import Env
from mindcraft.io import spaces
from typing import Optional, Tuple
from skimage.transform import resize
from numpy import argmax, ndarray, uint8
from collections.abc import Iterable


class GymWrapper(Env):
    """ Wrapper of popular OpenAI Gym environments to provide action-observation dynamics of the game-play to
        any agent. An environment can be chosen by passing the corresponding `gym_id` to the GymWrapper, see
        https://gym.openai.com/envs/#classic_control for possible environments.

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('gym_id', 'n_envs', 'reward', 'steps', 'gym_kwargs', 'seed', *Env.REPR_FIELDS, )

    def __init__(self,
                 gym_id: str = "CartPole-v1",
                 n_envs: int = 1,
                 reward: float = 0.,
                 steps: int = 0,
                 observation_space: Optional[spaces.Space] = None,
                 gym_kwargs: Optional[dict] = None,
                 seed: Optional[int] = None,
                 **kwargs):
        """ Constructs a GymWrapper instance

        :param gym_id: OpenAI Gym id (str), defaults to CartPole-v1, checkout https://gym.openai.com/envs/#classic_control.
        :param n_envs: Number of environments for potential vectorization using  (if greater than 1, defaults to 1).
        :param reward: Reward to start with, helper for bookkeeping, defaults to 0.
        :param steps: Initial value for counter for call of steps method.
        :param observation_space: Optional observation-space (mindcraft.io.spaces.Space) definition, defaults to
                                  specified `gym_id`'s observation-space.
        :param gym_kwargs: Optional kwarg-dictionary to be passed to Stable-Baseline-3's `make_vec_env`
                               util function.
        :param seed: Optional integer for seeding the environment.
        :param **kwargs:
        """

        self.gym_id = gym_id
        self.gym_kwargs = gym_kwargs or {}
        self._seed = seed

        assert n_envs >= 1
        if n_envs > 1:
            if make_vec_env is None:
                raise AttributeError("Issues with `from stable_baselines3.common.env_util import make_vec_env`, "
                                     "check `cv2` installation.")
            self.gym_env = make_vec_env(gym_id, n_envs=n_envs, **self.gym_kwargs)
            self.n_envs = n_envs
        else:
            self.gym_env = gym.make(gym_id, **self.gym_kwargs)
            self.n_envs = 1

        if observation_space is None:
            observation_space = self.gym_env.observation_space

        kwargs['omit_default'] = kwargs.get('omit_default', True)
        Env.__init__(self, observation_space=observation_space, reward=reward, steps=steps, **kwargs)
        self.seed(self._seed)

    def to_dict(self):
        dict_repr = Env.to_dict(self)
        if 'observation_space' in dict_repr:
            dict_repr['observation_space'] = repr(dict_repr['observation_space'])

        if 'seed' in dict_repr:
            dict_repr['seed'] = self._seed

        return dict_repr

    def __str__(self):
        return f"{self.gym_id}"

    def reset(self, action: object = None) -> object:
        self.steps = 0
        r = self.gym_env.reset()
        self.seed(self._seed)

        info = {}
        if isinstance(r, tuple):
            r, info = r

        return r, 0, False, info

    def seed(self, seed=None):
        if seed:
            self.gym_env.seed(seed)

    def step(self, action: object) -> Tuple[ndarray, float, bool, dict]:
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            if isinstance(action, (ndarray, Iterable)):
                if self.n_envs == 1 or np.ndim(action) > 1:
                    action = np.asarray(action)
                    if all(np.isnan(action)):
                        action[:] = 1.

                    elif any(np.isnan(action)):
                        action[np.isnan(action)] = -np.inf

                    p = np.array(softmax(action, axis=-1), dtype=np.float32)
                    action = np.random.choice(self.action_space.n, size=self.n_envs, p=p)[0]

        results = self.gym_env.step(action)
        try:
            observation, reward, done, truncated, info = results
        except ValueError:
            truncated = False
            observation, reward, done, info = results

        done |= truncated  # truncated if num_steps exceeds max_steps
        self.steps += 1

        if self.verbose:
            print(f'step {self.steps}, obs {observation.shape}, act {action.shape}, rew {reward}, done {done}, {info}')

        return observation, reward, done, info

    def render(self, *args, **kwargs):
        return self.gym_env.render(*args, **kwargs)

    def close(self):
        return self.gym_env.close()

    @property
    def action_space(self):
        return self.gym_env.action_space

    def sample_action_space(self):
        action_space = self.action_space
        if self.n_envs == 1:
            return action_space.sample()

        return [action_space.sample() for _ in range(self.n_envs)]

    def details(self):
        print("environment details:")
        print("- gym-id", self.gym_id)
        print("- env.action_space", self.action_space)
        try:
            print("  high, low", self.action_space.high, self.action_space.low)
        except AttributeError:
            pass

        print("- env.observation_space", self.gym_env.observation_space)
        try:
            print("  high, low", self.observation_space.high, self.observation_space.low)
        except AttributeError:
            pass

        try:
            print("- screen-size", self.gym_env.render(mode='rgb_array').shape)
        except AttributeError:
            pass

        return


class GymRGBWrapper(GymWrapper):
    """ Wrapper of popular OpenAI Gym environments to provide **pixel(RGB)**-observations of the game-play **screen** to
        any agent. An environment can be chosen by passing the corresponding `gym_id` to the GymRGBWrapper, see
        https://gym.openai.com/envs/#classic_control for possible environments.

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('screen_shape', 'screen_slice', 'channel_bias', 'channel_scale', 'invert', 'show_mode',
                   *GymWrapper.REPR_FIELDS, )

    SCREEN_X = 64
    SCREEN_Y = 64

    def __init__(self,
                 gym_id: str = "CartPole-v1",
                 screen_shape: tuple = (SCREEN_X, SCREEN_Y),
                 screen_slice: Optional[tuple] = None,
                 channel_bias: float = 0.,
                 channel_scale: float = 1.,
                 reward: float = 0.,
                 steps: int = 0,
                 invert: bool = False,
                 show_mode: Optional[str] = None,
                 **kwargs):
        """ Constructs a GymWrapper instance

        :param gym_id: OpenAI Gym id (str), defaults to CartPole-v1, checkout https://gym.openai.com/envs/#classic_control.
        :param screen_shape: Vertical and horizontal shape of the observation space (i.e., of the sliced game-screen),
                             Reshaped (i.e. compressed) image is the env. observation provided to the agent.
        :param screen_slice: Tuple of slices (vertical and horizontal directions) which are used to extract observation from full game-screen.
        :param channel_bias: Number subtracted from the pixel-wise values of the environment observation (`obs - bias`), defaults to 0..
        :param channel_scale: Scaling of the biased observation (`scale * (obs - bias)`), defaults to 1.
        :param reward: Reward to start with, helper for bookkeeping, defaults to 0.
        :param steps: Initial value for counter for call of steps method.
        :param invert: Flag to invert the observation (`1. - obs), applied prior to `bias` and `scale` transforms, defaults to False.
        :param show_mode: `'obs'`, `'frame'` or `None` to either display the (sliced, reshaped and transformed)
                          `observation` of the environment, the game-screen as-is, or Nothing at all (default).
                          This operation is performed at each call of the `step` method and allows to sneak-peak at
                          agent observations using `matplotlib.imshow`.
        :param **kwargs:
        """

        self.screen_shape = tuple(screen_shape)
        self.screen_slice = screen_slice
        self.channel_scale = channel_scale
        self.channel_bias = channel_bias
        self.invert = invert

        assert show_mode in ('obs', 'frame', None)
        self.show_mode = show_mode
        self.show_buffer = []

        # helpers
        self.obs_slice = {}

        # super
        observation_space = spaces.box.Box(low=0, high=255, shape=(*self.screen_shape, 3), dtype=uint8)
        if kwargs.get('n_envs', 1) > 1:
            raise NotImplementedError("``n_envs > 1 `` for RGB-Wrapper can't handle `process_frame` of several envs.")
        GymWrapper.__init__(self, gym_id=gym_id, reward=reward, steps=steps, observation_space=observation_space,
                            **kwargs)

    def reset(self, action: object = None) -> object:
        self.steps = 0
        self.gym_env.reset()

        if self.show_buffer:
            while self.show_buffer:
                import matplotlib.pyplot as plt
                import matplotlib.animation as animation
                import time

                fig = plt.figure()
                frames = [[plt.imshow(self.show_buffer[i], animated=True)] for i in range(len(self.show_buffer))]
                ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=500)
                ani.save(f'{self.gym_id}-{self.show_mode}-{str(time.time())}.mp4')
                plt.show()

            self.show_buffer = []

        return self._get_frame(), 0, False, {}

    def step(self, action: object) -> Tuple[ndarray, float, bool, dict]:
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            if isinstance(action, (ndarray, Iterable)):
                if self.n_envs == 1 or np.ndim(action) > 1:
                    action = argmax(action, axis=-1)

        observation, reward, done, truncated, info = self.gym_env.step(action)
        observation = self._get_frame()
        done |= truncated  # truncated if num_steps exceeds max_steps
        self.steps += 1

        if self.verbose:
            print(f'step {self.steps}, obs {observation.shape}, act {action.shape}, rew {reward}, done {done}, {info}')

        return observation, reward, done, info

    def _get_frame(self):
        try:
            frame = self.gym_env.render(mode='rgb_array')
        except TypeError:
            # assert self.gym_env.render_mode == 'rgb_array'
            frame = self.gym_env.render()

        return self._process_frame(frame)

    def _get_slice(self, frame):
        if self.screen_slice is None:
            return frame

        if self.obs_slice in ({}, None):
            if isinstance(self.screen_slice, slice):
                slice_x = self.screen_slice,
                slice_y = slice_x

            else:
                assert np.ndim(self.screen_slice) in (1, 2), "Either same x/y-slices or specific slices for x and y."

                if np.ndim(self.screen_slice) == 1:
                    slice_x = slice(*self.screen_slice)
                    slice_y = slice_x

                else:
                    slice_x, slice_y = self.screen_slice
                    slice_x = slice_x if isinstance(slice_x, slice) else slice(*slice_x)
                    slice_y = slice_y if isinstance(slice_y, slice) else slice(*slice_y)

            self.obs_slice = {'x': slice_x, 'y': slice_y}

        return frame[self.obs_slice['x'], self.obs_slice['y']]

    def _process_frame(self, frame):
        obs = self._get_slice(frame)
        obs = obs.astype(float) / 255.0
        obs = resize(obs, self.screen_shape)

        if self.invert:
            obs = 1. - obs

        if self.channel_bias != 0. or self.channel_scale != 1.:
            obs = (obs - self.channel_bias) * self.channel_scale

        if self.show_mode:
            self.plot_obs(frame, obs)

        return obs

    def plot_obs(self, frame, obs):
        self.show_buffer.append(obs if self.show_mode == 'obs' else frame)


class LunarLander(GymRGBWrapper):
    """ An RGB Wrapper for gym's LunarLander Environment.

    (c) B. Hartl 2021
    """
    def __init__(self,
                 screen_shape=(GymRGBWrapper.SCREEN_X, GymRGBWrapper.SCREEN_Y),
                 channel_bias=0.,
                 channel_scale=1.,
                 reward: float = 0.,
                 steps: int = 0,
                 invert: bool = False,
                 **kwargs):

        kwargs['gym_id'] = "LunarLander-v2"
        GymRGBWrapper.__init__(self, screen_shape=screen_shape,
                               channel_bias=channel_bias, channel_scale=channel_scale,
                               reward=reward, steps=steps, invert=invert, **kwargs)


class CartPole(GymRGBWrapper):
    """ An RGB Wrapper for gym's CartPole Environment.

    (c) B. Hartl 2021
    """

    def __init__(self,
                 screen_shape=(GymRGBWrapper.SCREEN_X, GymRGBWrapper.SCREEN_Y),
                 channel_bias=0.,
                 channel_scale=1.,
                 reward: float = 0.,
                 steps: int = 0,
                 invert: bool = False,
                 **kwargs):

        kwargs['gym_id'] = "CartPole-v0"
        GymRGBWrapper.__init__(self, screen_shape=screen_shape,
                               channel_bias=channel_bias, channel_scale=channel_scale,
                               reward=reward, steps=steps, invert=invert, **kwargs)


class CarRacing(GymRGBWrapper):
    """ An RGB Wrapper for gym's CarRacing Environment.

    (c) B. Hartl 2021
    """

    def __init__(self,
                 screen_shape=(GymRGBWrapper.SCREEN_X, GymRGBWrapper.SCREEN_Y),
                 channel_bias=0.,
                 channel_scale=1.,
                 reward: float = 0.,
                 steps: int = 0,
                 invert: bool = False,
                 **kwargs):

        kwargs['gym_id'] = "CarRacing-v2"
        GymRGBWrapper.__init__(self, screen_shape=screen_shape,
                               channel_bias=channel_bias, channel_scale=channel_scale,
                               reward=reward, steps=steps, invert=invert, **kwargs)


def random_rgb_env(env="LunarLander-v2", n_envs=1):
    """ A random-agent test of the `RGBWrapperEnv`

    (c) B. Hartl 2021
    """
    # Each of this episode is its own game.
    env = GymRGBWrapper(gym_id=env, n_envs=n_envs)
    env.details()

    env.reset()
    envs_done = np.zeros(env.n_envs, dtype=bool)
    while not all(envs_done):
        # This will display the environment
        # Only display if you really want to see it.
        # Takes much longer to display it.
        env.render(mode="human")

        # This will just create a sample action in any environment.
        # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
        action = env.sample_action_space()

        # this executes the environment with an action,
        # and returns the observation of the environment,
        # the reward, if the env is over, and other info.
        next_state, reward, done, info = env.step(action)

        # lets print everything in one line:
        if n_envs > 1:
            [print(f"\rEpisode {e}: ", env.steps, next_state[e].shape, reward[e], done[e], action[e], info[e])
             for e in range(n_envs)
             if not envs_done[e]]

        else:
            print(f"\rEpisode data: ", env.steps, next_state.shape, reward, done, action, info, )

        envs_done |= done


def random_env(env="LunarLander-v2", n_envs=1, render=False):
    """ A random-agent test of the `RGBWrapperEnv`

    (c) B. Hartl 2021
    """
    # Each of this episode is its own game.
    env = GymWrapper(gym_id=env, n_envs=n_envs)
    env.details()

    env.reset()
    # this is each frame, up to 500...but we wont make it that far with random.
    envs_done = np.zeros(n_envs, dtype=bool)
    while not all(envs_done):
        # This will display the environment
        # Only display if you really want to see it.
        # Takes much longer to display it.
        if render:
            env.render()  # mode="human")

        # This will just create a sample action in any environment.
        # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
        action = env.sample_action_space()

        # this executes the environment with an action,
        # and returns the observation of the environment,
        # the reward, if the env is over, and other info.
        next_state, reward, done, info = env.step(action)

        # lets print everything in one line:
        if n_envs > 1:
            [print(f"\rEpisode {e}: ", env.steps, next_state[e].shape, reward[e], done[e], action[e], info[e])
             for e in range(n_envs)
             if not envs_done[e]]

        else:
            print(f"\rEpisode data: ", env.steps, next_state.shape, reward, done, action, info, )

        envs_done |= done


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([random_rgb_env, random_env])
