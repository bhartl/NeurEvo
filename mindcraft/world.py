from numpy import abs as np_abs
from numpy import ndarray
from datetime import datetime
from time import sleep
from mindcraft.util.time import get_timestamp
from mindcraft.io import Repr
from mindcraft.io import Log
from mindcraft.io import vprint
from mindcraft import Env
from mindcraft import Agent
from typing import Union, Optional


class World(Repr, Log):
    """ The World class represents the stage where agents can interact with their environments.

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('agent', 'env',
                   'n_episodes', 'verbose', 'render', 'render_kwargs', 'max_steps', 'load_steps', 'load_action',
                   'log_fields', 'log_foos', 'delay', 'loss_factors',
                   'schedule',
                   'early_stopping', )

    def __init__(self,
                 agent: Union[Agent, dict],
                 env: Union[Env, dict],
                 n_episodes: int = 10,
                 verbose: bool = False,
                 render: bool = True,
                 render_kwargs: Optional[dict] = None,
                 max_steps: Union[int, None] = None,
                 load_steps: int = 0,
                 load_action: bool = False,
                 log_fields: Union[tuple, list] = Log.DEFAULT_LOG_FIELDS,
                 log_foos: (dict, None) = None,
                 schedule: (dict, None) = None,
                 delay: float = 0.,
                 early_stopping: Union[None, float] = None,
                 loss_factors: Optional[Union[tuple, list, dict]] = ()
                 ):
        """ Constructs a World instance

        :param env: Environment representable subjected to the agent.
        :param agent: Agent representable used to navigate in the environment
        :param n_episodes: Integer number of consecutive episodes when e.g. `make_rollouts` is called.
        :param verbose: Boolean (or integer) specifying whether (or in which intervals) rollout information is printed
                        (defaults to False).
        :param render: Boolean specifying whether the environment is rendered or not (defaults to True)
        :param render_kwargs: Optional kwargs for environment's render method, defaults to None.
        :param max_steps: Maximum number of steps until a rollout is stopped (defaults to None, i.e., no max-step
                          limit)
        :param: load_steps: Number of steps where no actions are applied and no logging is performed - this might be
                            necessary for some environments, which have a load-screen etc. (defaults to 0).
        :param log_fields: tuple or list of strings, specifying
                           (i) variables in the rollout method (i.e., 'observation', 'action', 'reward', 'loss', 'rating', 'info'),
                           (ii) properties or fields in the `World` instance (e.g., 'episode', 'agent', ...)
                           (iii) sub-properties of properties or fields in the `World` instance (e.g., 'agent.hidden_state', ...)
                           (iii) or keys in the `log_foos` dictionary,
                           which should be logged in a dictionary for each rollout.
                           One dict with lists for each log-field is generated per rollout.
                           The logs can be retrieved via the world's `log_history` property
        :param log_foos: key - value pairs which specify additional log-functions, where the values are either
                         `callable`s or string-code-snippets which can be subjected to the `compile` function.
                         If a log-field (in the `log` argument) corresponds to a key in the `log_foo` dict, the
                         value is executed (if it is callable) or subjected to `eval` otherwise.
                         The result is stored in the world's `log` property.
                         Such a log-foo key value pair could be `steps="self.env.steps"``, to log the step counts
                         of the environment.
        :param delay: Delay between rollout steps in seconds, e.g. to slow down rendering (defaults to 0.)
        :param schedule: An optional dictionary specifying a schedule for the world's parameters of the form
                         `setter-function(self, value)`: Union[`single value`, `list of values`] for each rollout step.
                         The `setter-function` is a method of the `World` class, or a function with the first
                         argument `self` (i.e., the `World` instance) and the second argument `value` (i.e., the
                         value of the schedule at the current rollout step) which implements to set the world's
                         attribute to the specified value, for example: `{lambda world, v: world.env.attr = v}`.
        :param early_stopping: Optional float, specifying a lower bound of successive negative cumulative reward when
                               the agent should stop (defaults to None, i.e. no early-stopping).
        :param loss_factors: Optional float or tuple of floats or dictionary {critic: loss} (and Nones)
                             to update the `loss_factor`s of the `Agent`'s `Critic`s, defaults to None.
                             If a single float is provided, all `Critic`s' `loss_factor''s are updated with that value.
                             If a tuple/list is provided, only floats are considered for updating values,
                             while Nones are used as placeholders to retain the corresponding `Critic`'s `loss_factor`s.
        """

        Repr.__init__(self, repr_fields=self.REPR_FIELDS, omit_default=True)
        Log.__init__(self, log_fields=log_fields, log_foos=log_foos)

        self._agent = None
        self._env = None
        self.agent = agent
        self.loss_factors = loss_factors
        self.env = env

        self.episode = 0
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.load_steps = load_steps
        self.load_action = load_action
        self.log_episode = None
        self.delay = delay

        self.schedule = schedule
        self._schedule = {}

        self.early_stopping = early_stopping
        self._negative_cum_reward = 0.

        self.verbose = verbose
        self.render = render
        self.render_kwargs = render_kwargs or {}

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = Env.make(value)

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = Agent.make(value)

    @property
    def loss_factors(self):
        return self.agent.loss_factors

    @loss_factors.setter
    def loss_factors(self, value):
        self.agent.loss_factors = value

    def vprint(self, *args, step=0, **kwargs):
        return vprint(*args, verbose=self.verbose, step=step, **kwargs)

    def apply_schedule(self, step):
        """ Applies schedule to world's parameters.

        :param step: Integer specifying the current rollout step.
        """
        if self.schedule is not None:
            for k, v in self.schedule.items():
                if k not in self._schedule:
                    schedule_foo = k
                    if not hasattr(schedule_foo, '__call__'):
                        try:
                            schedule_foo = compile(k, "<string>", "eval")

                        except:
                            # Compile the function string into a code object
                            compiled_code = compile(k, "<string>", "exec")

                            # Create an empty namespace to store the function
                            namespace = {}

                            # Execute the compiled code in the empty namespace
                            exec(compiled_code, namespace)

                            # Retrieve the function object from the namespace and store it in a variable
                            schedule_foo = [foo for k, foo in namespace.items() if callable(foo)][0]

                    self._schedule[k] = schedule_foo

                step_value = v
                if isinstance(v, (tuple, list, ndarray)):
                    step_value = v[step]
                self._schedule[k](self, step_value)

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        if self.schedule is not None:
            dict_repr["schedule"] = {k: self.v_to_list(v) for k, v in self.schedule.items()}
        return dict_repr

    def __enter__(self):
        return self.make(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.env.close()
        except:
            print('could not close environment.')
            pass

        try:
            self.agent.env.close()
        except:
            pass

    def get_parameters(self):
        return self.agent.get_parameters()

    def set_parameters(self, parameters):
        self.agent.set_parameters(parameters)

    def reset(self, action=None):
        self._negative_cum_reward = 0.

        # pybullet requires render before reset, gym can be the other way round
        render_on_reset = self.env.render_on_reset
        try:
            assert render_on_reset
            self.call_render()
            render_on_reset = False  # successfully rendered, e.g., pybullet
        except:  #  (AssertionError, AttributeError):
            render_on_reset = True

        finally:
            observation, reward, done, info = self.env.reset(action=action)
            if render_on_reset:
                self.call_render()

        Agent.reset_base(self.agent)
        action = self.agent.reset(observation, reward, info)
        if action is None:
            action = self.agent.get_default_action()

        return (observation, reward, done, info), action

    def call_render(self):
        if self.render:
            return self.env.render(**self.render_kwargs)

    def reset_log(self):
        Log.reset_log(self)
        self.episode = 0
            
    def rollout(self):
        """ A rollout of one episode of the environment """

        start = datetime.now()
        self.vprint(f"Rollout episode {self.episode + 1}/{self.n_episodes} at {start.strftime('%Y.%m.%d, %H:%M:%S')}")
        self.log_episode = self.episode

        steps = 0
        rating, loss = 0., 0.

        self.apply_schedule(steps)
        (observation, reward, done, info), action = self.reset()

        if self.load_steps == 0:
            self.log(observation=observation, reward=reward, done=done, info={},
                     action=action, loss=loss, rating=rating)

        # MAIN ACTION - OBSERVATION LOOP
        while not done:
            # AGENT PROPOSES ACTION BASED ON OBSERVATION
            is_loaded = self.is_loaded(steps)
            if is_loaded or self.load_action:
                action = self.agent.get_action(observation, reward, info)

            # UPDATE ENVIRONMENT BY AGENT'S ACTION
            observation, reward, done, info = self.env.step(action)
            self.call_render()

            # GET REWARD / LOSS RATING
            rating, loss = 0., 0.
            if is_loaded or done:
                rating, loss = self.agent.get_step_loss(observation=observation, reward=reward, action=action)

            # CHECK IF DONE
            steps += 1
            done |= self.is_done(is_loaded, steps, rating)

            # LOGGING
            if is_loaded:
                self.log(observation=observation, reward=reward, done=done, info={},
                         action=action, loss=loss, rating=rating)

            self.vprint(f"\rstep {steps}{f'/{self.max_steps}'*(self.max_steps is not None)}, "
                        f"total reward: {self.agent.total_reward}, "
                        f"total loss: {self.agent.total_loss}, "
                        f"total rating: {self.agent.total_rating}     ",
                        end='', step=steps-1)

            if not done:
                self.apply_schedule(steps)

            if self.delay != 0.:
                sleep(self.delay)

        episode_rating, episode_loss = self.agent.get_episode_loss()
        self.vprint(f"\rstep {steps}{f'/{self.max_steps}'*(self.max_steps is not None)}, "
                    f"total reward: {self.agent.total_reward}, "
                    f"episode loss: {episode_loss}, "
                    f"episode rating: {episode_rating}     ")
        dt = (datetime.now()-start).seconds
        self.vprint("Episode {} finished after {} time-steps ({} seconds).".format(self.episode, steps, dt))

        self.episode += 1
        if self.log_history and self.log_history[-1]:
            return self.wrap_np(self.log_history[-1])

        return episode_rating

    def is_loaded(self, steps):
        return self.load_steps <= steps

    def is_done(self, loaded, steps, reward):
        if self.max_steps is not None and steps >= self.max_steps:
            return True

        if loaded and self.early_stopping:
            if reward < 0.:
                self._negative_cum_reward += reward
            elif reward == 0. and self.early_stopping > 0:
                self._negative_cum_reward -= 0.1
            else:
                self._negative_cum_reward = 0.

            if self._negative_cum_reward < -np_abs(self.early_stopping):
                return True

        return False

    @staticmethod
    def make_rollouts(world, parameters=None, output_path=None, verbose=True, **world_kwargs):
        """ Static method do perform world rollout from world representable

        :param world: world object or list of world objects.
        :param parameters: Optional list of parameters which can be set via `world.set_parameters`, defaults to `None`.
        :param output_path: File-path for rollout results, written in hdf5 format (defaults to None).
        :param verbose: Boolean specifying whether status messages are printed to the screen, defaults to True.
        :param world_kwargs: Keyword-value pairs to be passed to `World.make` which will override world's
                             attributes. Could also include `partial_local`, i.e., a mapping of
                             {class-name: class-location/module} for representation make method, defaults to {}.
                             If multiple worlds are defined, `world_kwargs` can be defined in an appropriately
                             nested fashion: kwargs for the i-th world representable can be provided via the key
                             `world_i_kwargs = {**kwargs}` (but this is optional for each world).
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                print(f'World.make_rollouts: Start at {get_timestamp()}.')

            # potentially a list of worlds is provided
            world_rollouts = []
            multi_world = not isinstance(world, (World, dict, str))
            if not multi_world:
                world = (world, )

            for i, world_obj in enumerate(world):
                kwargs = world_kwargs
                if multi_world and any(k.startswith("world_") and k.endswith("_kwargs") for k in world_kwargs.keys()):
                    kwargs = world_kwargs.get(f"world_{i}_kwargs", {})

                with World.make(world_obj, **kwargs) as w:
                    if parameters is not None:
                        w.set_parameters(parameters)

                    if not verbose:
                        w.verbose = False

                    episode_rollouts = []
                    for n in range(w.n_episodes):
                        rollout = w.rollout()
                        episode_rollouts.append(rollout)

                    world_rollouts.append(episode_rollouts)

                if not multi_world:
                    world_rollouts = world_rollouts[0]

            if verbose:
                print(f'World.make_rollouts: Finished with rollout evaluations at {get_timestamp()}.')

            if output_path is not None:
                if verbose:
                    print(f'World.make_rollouts: Saving log_history to file `{output_path}` at {get_timestamp()}.')

                World.dump_history(output_path, log_history=world_rollouts, exist_ok=True)

            if verbose:
                print(f'World.make_rollouts: Done at {get_timestamp()}.')

            return world_rollouts
