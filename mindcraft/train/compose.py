import numpy as np
from mindcraft.io import Repr
from mindcraft.io import Log
from mindcraft import World
from pydoc import locate
from inspect import getmembers, isclass, getfile


class Compose(Repr, Log):
    """

    """

    REPR_FIELDS = ('world', 'static_world',
                   'rollout_kwargs', 'rollout_aggregation',
                   'method', 'method_module', 'method_kwargs',
                   'regularizer', 'r_loss_factor',
                   'log_fields', 'log_foos',
                   )

    # checkpoint constants
    AGENT_POSTFIX = '.yml'
    CHECKPOINT_POSTFIX = '-ckpt.h5'
    DEFAULT_ROLLOUT_KWARGS = {'output_path': None, 'verbose': None}

    def __init__(self,
                 world: (World, dict), static_world: bool = False,
                 rollout_kwargs=None, rollout_aggregation=None,
                 method: str = None, method_module: str = None, method_kwargs=None,
                 regularizer: (str, None) = None, r_loss_factor: float = 1.,
                 log_fields=(), log_foos=(),
                 ):

        Repr.__init__(self, repr_fields=self.REPR_FIELDS)
        Log.__init__(self, log_fields=log_fields, log_foos=log_foos)

        self.world_repr = world
        self.static_world = static_world
        if not static_world:
            self.world = world  # World.make(world, partial_local=rollout_kwargs.get('partial_local', {}))
        else:
            self.world = World.make(world, partial_local=rollout_kwargs.get('partial_local', {}))

        self.rollout_kwargs = rollout_kwargs or self.DEFAULT_ROLLOUT_KWARGS
        self.rollout_aggregation = rollout_aggregation

        self.method = method
        self.method_module = method_module
        self.method_kwargs = method_kwargs or {}

        self.regularizer = regularizer
        self.r_loss_factor = r_loss_factor

    def run(self, method=None, method_module=None, method_kwargs=None, **kwargs):
        if method is not None:
            self.method = method

        if method_module is not None:
            self.method_module = method_module

        if method_kwargs is not None:
            self.method_kwargs = method_kwargs

        optimizer = self.load_method()

        if optimizer is not self:
            return optimizer.run(**kwargs)

        return self.evaluate(**kwargs)

    def load_method(self) -> 'Compose':
        if self.method is None:
            return self

        method_module = self.method_module if self.method_module is not None else 'mindcraft.train'

        cls = None
        for method in [self.method + '_method', self.method]:
            loaded_method = locate(method_module + '.' + method)
            if loaded_method is None:
                continue

            if loaded_method is not None and isinstance(loaded_method, Compose):
                cls = loaded_method
                break

            for module_name, module_member in getmembers(loaded_method, isclass):
                if module_member.__name__ == self.method:
                    cls = module_member
                    break

                if self.method in getfile(module_member):
                    cls = module_member
                    break

        if cls is not None:
            kwargs = {k: v for k, v in self.to_dict().items() if k not in ('method', 'method_module', 'method_kwargs')}
            kwargs = {**kwargs, **self.method_kwargs, 'class_': cls}
            return cls.make(kwargs)

        raise AttributeError(self.method_module, self.method)

    @classmethod
    def make_evaluate(cls, opt_repr, partial_local, parameters=None):
        optim = cls.make(opt_repr, partial_local=partial_local)
        return optim.evaluate(parameters=parameters)

    def evaluate(self, parameters=None):
        rollouts = World.make_rollouts(self.world, parameters=parameters, **self.rollout_kwargs)
        agg = self.aggregate(rollouts)

        if parameters is not None and np.isscalar(agg) and self.regularizer is not None:
            reg = self.r_loss(parameters)
            return agg - reg * self.r_loss_factor

        return agg

    def r_loss(self, parameters):
        if self.regularizer.lower() == 'l1':
            return np.mean(np.abs(parameters))

        elif self.regularizer.lower() == 'l2':
            return np.mean(np.abs(parameters)**2)

        raise NotImplementedError(self.regularizer)

    def aggregate(self, rollouts):
        if self.rollout_aggregation is None or np.isscalar(rollouts):
            return rollouts

        agg = self.rollout_aggregation
        agg = getattr(np, agg, getattr(locals(), agg, getattr(globals(), agg, agg)))

        if all([np.isscalar(r) for r in rollouts]):
            return agg(np.asarray(rollouts))

        if all([isinstance(rollout, list) for rollout in rollouts]):
            # list of lists -> several worlds with multiple episodes
            # sum over aggregation of different lists of ...
            try:
                # 1) assume r items are cumulative reward
                return np.sum([agg(np.asarray(r, dtype=float)) for r in rollouts])

            except TypeError:
                # 2) assume list of lists of episode_log-dicts
                return np.sum([
                    agg(np.asarray([np.sum(log['reward']) for log in r], dtype=float))
                    for r in rollouts
                ])

        if all([isinstance(rollout, dict) for rollout in rollouts]):
            # aggregation of several episodes -> list of dicts
            return agg([np.asarray(np.sum(r['reward'])) for r in rollouts])

        raise NotImplementedError('Do not understand rollouts format:', rollouts)

    @classmethod
    def x_from_history(cls, file, parse_log_file=True, run_key=-1, step_key=-1):
        if parse_log_file:  # parse to checkpoint file
            file = cls.get_checkpoint_file(file)

        import h5py
        with h5py.File(file, 'r') as h5:
            # pick run - key
            if int(run_key) < 0:
                run_key = str(list(sorted(h5.keys()))[int(run_key)])

            # load x value of specified run
            x = h5[run_key]['x']

            # pick optimization step
            if int(step_key) < 0:
                step_key = str(list(sorted(h5.keys()))[int(step_key)])

            if isinstance(x, np.ndarray):
                return x[int(step_key)]

            try:
                return np.asarray(x[str(step_key)])

            except AttributeError:
                return np.asarray(x[step_key])

    @classmethod
    def get_agent_file(cls, log_file):
        if log_file:
            agent_file = log_file

            if log_file.endswith('.log'):
                agent_file = agent_file.replace('.log', '')

            return agent_file + cls.AGENT_POSTFIX

        return None

    @classmethod
    def get_checkpoint_file(cls, log_file):
        if log_file:
            checkpoint_file = log_file

            if log_file.endswith('.log'):
                checkpoint_file = checkpoint_file.replace('.log', '')

            return checkpoint_file + cls.CHECKPOINT_POSTFIX

        return None
