"""
Collected functionality to work with an arbitrary mindcraft Agent instance
in an arbitrary gym environment.

Implements the following methods:

- `rollout` ... perform agent rollouts in the environment.
- `train`   ... train an agent using evolutionary strategies (ES).
                Use the --es-util [CMAES, PEPG, SimpleGA, OpenES] option, to pick a certain algorithm.

:examples:

Execute pretrained agent (from project root):

```python examples/agents/gym/gymcraft.py rollout --task classic_control/cart_pole --conf feed_forward ```

Where `--task` specifies the directory of the gym environment,
and `--conf` specifies the specific folder within the `--task/config/--conf` directory with all necessary definitions,
i.e.:

  - The `agent.yml` Agent definition file,
  - The `env.yml` Environment definition file,
  - The `fresh_world.yml` fresh world definition, connection the `agent.yml` and `env.yml` config files,
  - The `pretrained_world.yml` file for retraining, connecting the former solution `evolved-agent.yml` and `env.yml` config files,
  - And a `rollout.yml`, which is used to deploy the `evolved-agent.yml` in its `env.yml` with graphical output.

Train [Retrain] Agent on 16 cores via MPI on Ubuntu with an ES algorithm:

```mpirun -n 16 python examples/agents/gym/gymcraft.py train --task classic_control/cart_pole --conf feed_forward -s 96 -g 100 --checkpoint-interval 10 [--retrain]```

(c) B. Hartl 2021
"""

from typing import Union, Optional
import os


def merge_path(path, prefix):
    while prefix.endswith('/') and path.startswith('/'):
        path = path[1:]

    return os.path.join(prefix, path)


def get_config_path(config, prefix):
    if "config" not in config[len("config") + 1:]:
        config = merge_path(config, prefix="config")

    return merge_path(config, prefix=prefix)


def get_models_path(config, prefix):
    if "config" in config[len("config") + 1:]:
        config = config.replace("config", "models")
    else:
        config = merge_path(config, prefix="models")

    model_path = merge_path(config, prefix=prefix)
    if model_path.startswith('/'):
        return '/data' + model_path

    return 'data/' + model_path


def get_environment_yml(config, prefix):
    config = get_config_path(config, prefix=prefix)
    return merge_path("env.yml", prefix=config)


def get_agent_yml(config, prefix):
    config = get_config_path(config, prefix=prefix)
    return merge_path("agent.yml", prefix=config)


def get_fresh_world_yml(config, prefix):
    config = get_config_path(config, prefix=prefix)
    return merge_path("fresh_world.yml", prefix=config)


def get_pretrained_world_yml(config, prefix):
    config = get_config_path(config, prefix=prefix)
    return merge_path("pretrained_world.yml", prefix=config)


def get_rollout_yml(config, prefix):
    config = get_config_path(config, prefix=prefix)
    return merge_path("rollout.yml", prefix=config)


def train(task="classic_control/cart_pole",
          conf='feed_forward',
          file_name="evolved-agent.yml",
          prefix="examples/agents/gym",
          retrain=False,
          size=16,
          generations=30,
          aggregation='mean',
          opts: Union[None, dict, str] = None,
          checkpoint_interval: int = 0,
          es: str = 'CMAES',
          threshold: Optional[float] = None,
          verbose=True,
          workload_pool=False,
          ):
    """ Train a
        :class:`TorchAgent<mindcraft.agents.TorchAgent>`
        with an ES algorithm from
        :class:`mindcraft.train.es_utils<mindcraft.train.es_utils>`
        using the
        :func:`mindcraft.train<mindcraft.train>`
        function.

        See :func:`mindcraft.train<mindcraft.train>` for details on the parameters
        (can be accessed via the command-line help).

        Use the --es [CMAES, PEPG, SimpleGA, OpenES] option, to pick a certain algorithm.

        (c) B. Hartl 2021
    """
    from mindcraft import train as adapt
    if prefix not in ("", None):
        task = merge_path(task, prefix=prefix)

    if not retrain:
        world_repr = get_fresh_world_yml(config=conf, prefix=task)

    else:
        world_repr = get_pretrained_world_yml(config=conf, prefix=task)

    model_path = get_models_path(conf, task)

    return adapt(world_repr=world_repr,
                 opts=opts,
                 size=size,
                 generations=generations,
                 aggregation=aggregation,
                 verbose=verbose,
                 new_model=not retrain,
                 path=model_path,
                 file_name=file_name,
                 checkpoint_interval=checkpoint_interval,
                 es=es,
                 threshold=threshold,
                 async_workload=workload_pool,
                 )


def rollout(task="classic_control/cart_pole", conf='feed_forward', prefix="examples/agents/gym"):
    """ Run rollouts in environment of the specified agent
        via the
        :func:`mindcraft.rollout<mindcraft.rollout>`
        function.

        (c) B. Hartl 2021

    :param task: The agent's rollout configuration as dict-like or path, defaults to `ROLLOUT_CONFIG`.
    :param conf:
    :param prefix:
    :return: results from (:func:`mindcraft.rollout<mindcraft.rollout>`)
    """
    from mindcraft import rollout
    if prefix not in ("", None):
        task = merge_path(task, prefix=prefix)

    rollout_yml = get_rollout_yml(conf, prefix=task)
    print(f"rolling out: '{rollout_yml}'")
    return rollout(config=rollout_yml)


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([
        rollout,
        train
    ])
