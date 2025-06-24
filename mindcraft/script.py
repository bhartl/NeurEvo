""" Collection of scripts to interact with the `mindcraft` framework.

(c) B. Hartl 2021
"""
import os
from typing import Union, Optional
import glob


def rollout(config, **rollout_kwargs):
    """ Perform world rollouts based on the `mindcraft.adapt.Compose` `run` interface:
    a config representable (file, dict, ...) can be specified and additional `rollout_kwargs`.
    can be provided.

    This method can be used to save world rollout results to a specified `output_path` (specivied
     in `rollout_kwargs` either in the config or as kwarg) in hdf5-file format.

    See `mindcraft.compose.WorldOptimizer` for more details.

    :param config: `mindcraft.World` representation to compose a rollout configuration (dict or path to yml/json file)
    :param rollout_kwargs: Additional, optional keyword arguments for `mindcraft.adapt.Compose` instance's
                           `rollout_kwargs` argument.

    (c) B. Hartl 2021
    """
    from mindcraft.train import Compose
    compose_rollout = Compose.make(config)
    if rollout_kwargs:
        compose_rollout.rollout_kwargs = rollout_kwargs

    return compose_rollout.run()


def train(world_repr,
          static_world=False,
          es_method=None, es='CMAES', size=16, generations=30, aggregation='mean',
          verbose=1, new_model=False,
          path='data/examples/agents/models/',
          file_name='evolved.yml',
          reg: (str, None) = None,
          opts: Union[None, str, dict] = None,
          checkpoint_interval: int = 0,
          checkpoint=None,
          threshold: Optional[Union[int, float]] = None,
          dump_models: bool = False,
          dump_interval: bool = None,
          dump_world: Optional[str] = None,
          comm=None,
          async_workload=False,
          run_kwargs=None,
          **es_kwargs
          ):
    """ Training world parameters with evolutionary algorithms.

    The solution is exported to the specified joined `path` + `filename`,
    logs can be found in the corresponding `path` + `filename -> log_file`, where the postfix of the `filename` is
    replaced with '.log'.

    :param world_repr: Representation of World configuration (dict-like, path to, etc).
    :param static_world: Boolean, controlling whether to keep a 'static' `world` instance in the optimizer
                         via World.make(`world_repr`), defaults to `False`.
                         *Note:*
                         *This might be relevant for NCP routing optimization, to maintain routings over generations.*
    :param es_method: type, `mindcraft.train.EvolutionaryStrategy` Evolutionary Strategy Solver implementation,
                      defaults to None (which will load `EvolutionaryStrategy`).
    :param es: Evolutionary Strategy Solver implementation (see mindcraft.train.es_util.py),
                    choose among ('CMAES', 'SimpleGA', 'OpenES', 'PEPG').
    :param size: Population size of  evolutionary optimization.
    :param generations: Number of generations of evolutionary optimization.
    :param aggregation: Aggregation of fitness of different episodes (mean, min or max, ...).
    :param verbose:
    :param new_model: Boolean controlling, whether a fresh model is used for training, or if an existing one is used.
    :param path: Operating directory of simulation (log-file and yml-file destination).
    :param file_name: File-name of model output (yml) file.
    :param reg: weights regularizer, e.g. 'l1' or 'l2'.
    :param opts: Optional dict or dict-string representation of possible `inopt` parameters of the cma.CMAES optimizer,
                 an example would be `opts="{'CMA_elitist': 1, 'sigma_init': 0.5, 'weight_decay': 0.001}"`
    :param checkpoint: Tuple or str representation of `(log_file: str, runs: int = -1, gens: int = -1)` specifying the checkpoint to load.
    :param checkpoint_interval: Interval in which checkpoints of the trained agent are stored,
                                defaults to 0 (no checkpoints).
    :param threshold: Optional float, specifying collective reward (aggregated over episodes) when to stop optimization
                      if exceeded, defaults to `None`.
    :param dump_models: Boolean, specifying whether to dump only the model in the `trainables` list, or otherwise
                        the entire agent (per default if `False`).
    :param dump_interval: Optional integer, specifying optimization steps after which an agent-model is dumped to
                          the respective agent-file, defaults to None.
    :param dump_world: Optional filename to dump the world configuration which corresponds to the agent-file.
    :param comm: Optional `MPI.Comm(...)` instance, specifying the parallel resources available to the optimization
                 prozecudre (defaults to None, i.e., all resources are used. c.f. `MPI.COMM_WORLD.split(...)`,
                 to divide the world communicator into several colors:
                 https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html).
    :param async_workload: Flag to enable `workload.Pool` asynchronous parallelization for evaluating
                           a genertion's fitness score, defaults to False.
    :return: None

    (c) B. Hartl 2021
    """
    agent_file = ""
    if not os.path.isdir(path):
        if len(glob.glob(path + '*' * ('*' not in path))):
            agent_file = path + file_name
    agent_file = agent_file or os.path.join(path, file_name)
    if not "." in agent_file:
        agent_file += '.yml'

    log_file = '.'.join(agent_file.split('.')[:-1]) + '.log'

    # load potentially existing starting parameters
    from mindcraft.train.es_method import setup_optimizer
    optimizer_cls, x0, opts, kwargs = setup_optimizer(world_repr, es, new_model, opts=opts, comm=comm, es_method=es_method)
    kwargs = {**es_kwargs, **kwargs}

    if dump_world:
        from mindcraft import World
        world_dict = World.make(world_repr).to_dict()
        world_dict["agent"] = agent_file
        os.makedirs(os.path.dirname(dump_world), exist_ok=True)
        with open(dump_world, 'w') as f:
            import yaml
            yaml.safe_dump(world_dict, f, sort_keys=False)

    if checkpoint:
        if isinstance(checkpoint, str):
            import json
            checkpoint = json.loads(checkpoint)
        x0 = optimizer_cls.load_checkpoint(log_file, **checkpoint)

    # initialize solver
    optimizer = optimizer_cls(
        comm=comm,
        world=world_repr,  # [world_repr, world_repr],
        rollout_kwargs={'verbose': False, },
        rollout_aggregation=aggregation,
        population_size=size,
        static_world=static_world,
        x0=x0,
        log_file=log_file,
        regularizer=reg,
        checkpoint_interval=checkpoint_interval,
        opts=opts,
        threshold=float(threshold) if threshold else None,
        dump_models=dump_models,
        dump_interval=dump_interval,
        async_workload=async_workload,
        **kwargs
    )

    solution = None
    try:
        run_kwargs = run_kwargs or {}
        result = optimizer.run(max_steps=generations, verbose=verbose, **run_kwargs)

        if optimizer.is_root:
            solution = result['x']

    except KeyboardInterrupt:
        if optimizer.is_root:
            optimizer.persist('Aborted via Keyboard Interrupt.')

    except Exception as ex:
        if optimizer.is_root:
            optimizer.persist(f'{type(ex)} occurred:', str(ex))
        raise

    if optimizer.is_root and solution is not None:
        optimizer.dump(parameters=solution, file_name=agent_file, verbose=verbose)
        optimizer.persist('done')


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([
        rollout,
        train,
    ])
