import mpi4py.MPI
import torch
from mindcraft.train import Compose
from mindcraft import World
from collections import ChainMap
from datetime import datetime
import numpy as np
import os
from typing import Optional, Union
import h5py
import ast
import copy


class EvolutionaryStrategy(Compose):
    """ An ES Optimizer to tune the parameters of mindcraft.World instances/representations

    also see
        - https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
        - https://blog.otoro.net/2017/11/12/evolving-stable-strategies/

    taken from B. Hartl 2021
    """

    REPR_FIELDS = ('es_util', 'population_size', 'n_params', 'x0', 'log_file', 'verbose', 'log_fields',
                   'checkpoint_interval', 'opts', 'threshold', 'dump_models',
                   *Compose.REPR_FIELDS)

    LOG_FIELDS = ('x',           # best solution at time of checkpoint
                  'parameters',  # whole population's parameters at time of checkpoints
                  'reward',      # reward of entire population at time of checkpoints
                  'duration',    # duration of optimization step
                  )

    LOG_FOOS = None

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = (rank == 0)

    @classmethod
    def world_bcast(cls, value, root=0):
        return cls.comm.bcast(value, root=root)

    def bcast(self, value, root=0):
        return self.comm.bcast(value, root=root)

    def __init__(self,
                 es_util='CMAES',
                 population_size=101,
                 threshold: Optional[Union[int, float]] = None,
                 n_params: Optional[int] = None,
                 x0: Optional[Union[list, tuple, np.ndarray, torch.Tensor]] = None,
                 verbose: bool = True,
                 log_file: str = 'es.log',
                 log_fields: tuple = LOG_FIELDS,
                 log_foos: Optional[dict]  =LOG_FOOS,
                 checkpoint_interval: int = 0,
                 opts: Optional[Union[dict, str]] = None,
                 dump_models: bool = False,
                 dump_interval: Optional[int] = None,
                 comm: Optional[mpi4py.MPI.Comm] = None,
                 async_workload: bool = False,
                 diversity_metric: Optional[str] = "cdist",
                 **kwargs):
        """ Constructs a CMAESOptimizer instance

        :param population_size: population size of the CMA-ES algorithm (see es_util.py,
                                a CMA-ES instance is instantiated in run)
        :param threshold: Return (aggregated collective reward) threshold, signaling when to stop optimization.
        :param log_file: Filename to the log-file, defaults to `cma-es.log`.
        :param log_fields: List of field names to log after each generation (choose from LOG_FIELDS).
        :param log_foos: Dict which may hold log-instructions applied to the `log_fields`. If a key aligns
                         with a log-field, the value of will be subjected to `eval(value)`, and the result
                         will be logged.
        :param verbose: Boolean specifying whether the optimizer should print log-messages on the screen
                        (defaults to True).
        :param checkpoint_interval: Positive integer number, specifying the number of evolutionary steps after which
                                    a checkpoint of the current optimization process is dumped -- additional to when
                                    a new best solution is found, that's always being taken care of if a
                                    `log_file` is specified. Defaults to `0`, i.e., only new best solutions
                                    are subjected to the checkpoint policy.
        :param opts: Optional dictionary, specifying the options for the es-util algorithm. If a string is provided,
                     it is evaluated to a dictionary (defaults to None).
        :param dump_models: Boolean, specifying whether to dump only the model in the `unfreeze` list, or otherwise
                            the entire agent (per default if `False`).
        :param dump_interval: Optional integer, specifying after how many iterations of the es-solver the agent is dumped
                              to a file (if a new best agent is dumped, the counter starts from 0), defaults to None.
        :param kwargs: Forwarded to WorldOptimizer.__init__
        :param comm: Optional `MPI.Comm(...)` instance, specifying the parallel resources available to the optimization
                     preprocedure (defaults to None, i.e., all resources are used. c.f. `MPI.COMM_WORLD.split(...)`,
                     to divide the world communicator into several colors:
                     https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html).
        :param async_workload: Flag to enable `workload.Pool` asynchronous parallelization for evaluating
                               a genertion's fitness score, defaults to False.
        :param diversity_metric: Optional string from `("cdist", )`, specifying the diversity metric to be used for the ES algorithm.
        """

        Compose.__init__(self, log_fields=log_fields, log_foos=log_foos, **kwargs)
        self.population_size = population_size
        self.threshold = threshold

        if os.path.split(log_file)[0] not in ('', (), None):
            os.makedirs(os.path.split(log_file)[0], exist_ok=True)

        self.rollout_cycle = 0

        # setup logfile
        self.log_file = log_file
        self.verbose = verbose

        # setup checkpoint file
        self.checkpoint_interval = checkpoint_interval

        # define evolutionary solver from es-util
        self.es_util = es_util
        self.es = None

        # params init
        self.x0 = x0
        self.n_params = None
        self.set_nparams(n_params)

        # opt init
        self._opts = None
        self.opts = opts

        self.dump_models = dump_models
        self.dump_interval = dump_interval

        self.comm = None
        self.size = None
        self.rank = None
        self.is_root = False
        self.init_MPI(comm=comm)
        self.async_workload = async_workload

        self.diversity_metric = diversity_metric

        # helpers
        self.best_reward = -np.inf
        self.best_avg = -np.inf
        self.result = None

    def init_MPI(self, comm):
        """ Inits MPI communicator if `None` is provided, else uses the provided (potentially split-) communicator
            for multiprocessing. Sets `rank`, `size` and `is_root` properties of the es-method instance. """
        if comm is not None:
            self.comm = comm
        else:
            self.comm = EvolutionaryStrategy.comm

        self.size = self.comm.size
        self.rank = self.comm.rank
        self.is_root = (self.rank == 0)

    def set_nparams(self, n_params):
        if n_params is None:
            if self.x0 is not None:
                self.n_params = len(self.x0)

            else:
                with World.make(self.world) as w:
                    x = w.get_parameters()
                    if hasattr(x, '__len__'):
                        assert np.ndim(x) == 1
                        self.n_params = len(x)

        else:
            self.n_params = n_params

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, value):
        value = value or {}
        if isinstance(value, str):
            value = eval(value)

        self._opts = value or {}

    def show_opts(self, es_util=None):
        """ returns a dictionary with all possible options for the CMA-ES """
        if (es_util and 'cma' in es_util.lower()) or 'cma' in self.es_util:
            from cma.evolution_strategy import CMAOptions
            return CMAOptions()

        from .es_util import get_es_solver
        _, signature = get_es_solver(es_util if es_util else self.es_util, return_signature=True)
        return signature

    def persist(self, msg, rank=False, **kwargs):
        """ A helper function which prints a `msg` on the screen (if self.verbose) and
            which logs the `msg` as a line to the specified `self.log_file`.

        :param msg: String message to be logged.
        :param rank: Boolean specifying whether all ranks should print/persist to the log_file (if True),
                     or if only the main-thread (rank 0) should do this (defaults to False, i.e. only rank 0 logging)
        """

        if self.is_root or rank:
            if self.verbose:
                print(msg, **kwargs)

            with open(self.log_file, 'a') as s:
                end = kwargs.get('end', "\n")
                s.write(msg + end)

    def evolutionary_strategy(self, **kwargs):
        self.es_util = kwargs.pop('es_util', self.es_util)
        self.persist(f"loading es-util solver `{self.es_util}`")
        self.persist(f"solver options: {self.opts}")

        # MANAGE KWARGS IN OPTS
        from .es_util import get_es_solver
        cls, signature = get_es_solver(self.es_util, return_signature=True)
        signature_opts = {opt: self.opts.pop(opt) for opt in signature.parameters.keys() if opt in self.opts}

        # CREATE MANDATORY KWARGS
        es_kwargs = dict(
            num_params=kwargs.pop('n_params', self.n_params),
            popsize=kwargs.pop('size', self.population_size),
            x0=kwargs.pop('x0', self.x0)
        )

        # MERGE WITH OPTIONAL KWARGS
        es_kwargs = {**es_kwargs, **{opt_key: opt_value
                      for opt_key, opt_value in signature_opts.items()
                      if opt_key not in kwargs and opt_value is not None}}

        # CREATE EVOLUTIONARY SOLVER
        try:
            if "inopts" in signature.parameters.keys():
                es_kwargs['inopts'] = self.opts

            return cls(**es_kwargs, **kwargs)

        finally:
            for opt_key, opt_value in signature_opts.items():
                self.opts[opt_key] = opt_value

    def reset(self):
        self.best_reward = -np.inf
        self.best_avg = -np.inf
        self.result = None

    def run(self, max_steps=100, verbose=None, x0=None, **kwargs):
        """ Perform optimization with the cma-es optimization (see mindcraft.train.es_util.py file)

        :param max_steps: maximum number of generations of the evolutionary algorithm
        :param verbose: Boolean or integer specifying, whether verbose log-messages should be printed to the screen.
                        If an integer > 0 is specified, the log-messages will occur each `verbose` generations.
        :param x0: Optional list of initial parameters
        :returns: result dictionary with keys
                  'x' (the parameter solution),
                  'fitness' (the aggregated fitness of the solution),
                  'history' (a chronological fitness list over training steps).
        """
        # SETUP LOG AND CHECKPOINT HANDLING
        time_start = datetime.now()
        checkpoint_file = self.checkpoint_file
        checkpoint_key = True  # helper variable for finding run key for checkpoint hdf5 file

        if verbose is None:
            verbose = self.verbose

        # SETUP RESULT HANDLING
        x0 = x0 if x0 is not None else self.x0
        parameters = x0

        try:
            # load multiprocessing, if parallel environment is specified accordingly
            self.persist(F"ES: initializing")
            given_population = False

            # load ES solver from es-utils
            if self.is_root:
                if self.es is None:
                    self.es = self.evolutionary_strategy(x0=x0, **kwargs)

                elif x0 is not None:
                    given_population = len(x0) == self.population_size
                    if given_population:
                        self.es.ask()
                        self.es.flush(x0)

                    else:
                        self.es.inject(x0)

            # start optimization
            time_step = datetime.now()
            not_dumped_since = 0

            pool = None
            if self.async_workload:
                from workload import Pool
                pool = Pool(foo=self.evaluate,
                            comm=self.comm,
                            quiet=True,
                            logfile=None,
                            filename=None,
                            )

            for step in range(max_steps):

                # retrieve current solutions from es solver
                keys, parameters, solutions = None, None, None
                if self.is_root:
                    if not given_population or step > 0:
                        parameters = self.es.ask()
                    else:
                        parameters = x0

                    agent_parameters = self.get_agent_parameters(parameters, allow_mutable=True)
                    if step == 0:
                        n_cores = self.size
                        self.persist(f"ES: starting {self.es.__class__.__name__} optimizer "
                                     f"on {n_cores} core{'s' * (n_cores > 1)}" +
                                     f", distributed on a `workload.Pool`" * self.async_workload)
                        self.persist(f"population-size: {len(parameters)}")
                        if self.n_params:
                            msg = f"with num. parameters: {self.n_params}"
                            if agent_parameters.shape[-1] != self.n_params:
                                msg += f" (encoding {agent_parameters.shape[-1]} agent parameters)"
                            self.persist(msg)
                    # parameters = agent_parameters

                if not self.async_workload:
                    if self.is_root:
                        keys = np.array_split(range(len(agent_parameters)), self.size)
                        solutions = np.array_split(agent_parameters, self.size)
                    keys = self.comm.scatter(keys, root=0)
                    solutions = self.comm.scatter(solutions, root=0)

                    try:
                        self.world.reset_log()
                    except:
                        pass

                    # serial or mpi evaluation
                    reward_list = {k: self.evaluate(s) for k, s in zip(keys, solutions)}

                    # gather evaluations (list of dicts with keys representing solution index)
                    reward_list = self.comm.gather(reward_list, root=0)
                    if self.is_root:
                        # merge list of dicts into single dict with integer keys for all solutions and make list
                        reward_list = dict(ChainMap(*reward_list))
                        reward_list = np.asarray(self.dict_to_list(reward_list))

                    # gather world log history and sort by keys (workload -> genotype -> episode mapping)
                    if isinstance(self.world, World):
                        log = {k: self.world.log_history[i * self.world.n_episodes:(i + 1) * self.world.n_episodes]
                               for i, k in enumerate(keys)}
                        log = self.comm.gather(log , root=0)
                        if self.is_root:
                            log = dict(ChainMap(*log))
                            log = sum([log[k] for k in sorted(log.keys())], [])
                            self.world.log_history = log

                            try:
                                shapes = {k: log[0][k].shape for k in self.world.log_fields}
                                reshape = lambda k: (self.es.popsize, self.world.n_episodes, *shapes[k])
                                es_log = {}
                                for k in self.world.log_fields:
                                    vk = [v[k] for v in log]
                                    es_log[k] = [vk[i*self.world.n_episodes:(i+1)*self.world.n_episodes]
                                                 for i in range(self.es.popsize)
                                                 ]

                                    try:
                                        es_log[k] = np.reshape(es_log[k], reshape(k))
                                    except:
                                        pass

                                self.es.world_log = es_log
                            except:
                                pass

                        else:
                            self.world.reset_log()

                else:
                    reward_list = pool.execute_workload(workload_args=agent_parameters, drop_previous=True, workload_sort=True)
                    if self.is_root:
                        reward_list = np.asarray(reward_list['foo_result'])

                done = False
                if self.is_root:
                    # send fitness list to ES algorithm
                    self.es.tell(reward_list)

                    # extract results and log
                    current_reward = np.max(reward_list)
                    current_avg = np.mean(reward_list)
                    if self.threshold and current_reward >= self.threshold:
                        done = True

                    # evaluate duration for logging
                    duration = (datetime.now() - time_step).total_seconds()

                    # checkpoint handling
                    new_best = current_reward > self.best_reward or (current_reward >= self.best_reward and current_avg > self.best_avg)
                    self.best_reward = max([self.best_reward, current_reward])
                    self.best_avg = max([self.best_avg, current_avg])

                    dump_checkpoint = bool(self.checkpoint_interval) and not (step+1) % self.checkpoint_interval  # by interval
                    dump_checkpoint |= new_best
                    dump_checkpoint &= bool(checkpoint_file)

                    if new_best or (self.dump_interval and (not_dumped_since + 1) % self.dump_interval):
                        self.result = self.es.result()  # best_x, best_fitness, best_fitness, sigma
                        self.dump(parameters=self.result[0], file_name=self.agent_file, verbose=False)
                        not_dumped_since = 0

                    else:
                        not_dumped_since += 1

                    if dump_checkpoint or self.checkpoint_interval < 0:
                        self.log(x=self.result[0], parameters=agent_parameters, reward=reward_list, duration=duration,
                                 episode_key=step)

                    if dump_checkpoint:
                        checkpoint_key = self.dump_history(filename=checkpoint_file, exist_ok=True,
                                                           log_history=[self.log_history[-1]],  # only dump most recent
                                                           key_offset=checkpoint_key,  # specifies run-specific key
                                                           )
                        # reset history -> otherwise memory grows
                        self.log_history = []

                    # print log message
                    if verbose and (step == 0 or (step + 1) % verbose == 0):
                        diversity_msg = ""
                        if self.diversity_metric == "cdist":
                            params = agent_parameters
                            diversity = [np.linalg.norm(params[i] - params[j]) for i in range(len(params)) for j in range(i + 1, len(params))]
                            diversity_mean = np.mean(diversity)
                            diversity_std = np.std(diversity)
                            diversity_msg = f"Diversity: {diversity_mean:.3f}({diversity_std:.3f}), "

                        msg = f"Iteration {(step + 1)}/{max_steps} -> " \
                              f"Best Reward: {self.best_reward:+.3f}, " \
                              f"Current Reward: {current_reward:+.3f}, " \
                              f"Avg(Std)-Reward: {current_avg:+.3f}({reward_list.std():.3f}), " + diversity_msg + \
                              f"Duration: {duration:.2f} secs."

                        self.persist(msg)

                    # reset timer
                    time_step = datetime.now()

                if self.bcast(done, root=0):
                    if self.is_root:
                        self.persist(f'ES: reward-threshold of {self.threshold} exceeded, stopping.')
                    break

        except Exception as ex:
            self.persist(f"{type(ex)}:{str(ex)}")
            raise

        if verbose and self.is_root:
            x = self.result[0]
            if isinstance(x, np.ndarray):
                x = x.tolist()

            self.persist(f"ES: local optimum discovered by solver, with score: {self.result[1]}.")
            if self.checkpoint_file:
                self.persist(f'ES: logger-history @ `{self.checkpoint_file}`.')

        if self.is_root and self.result is not None:
            return dict(x=self.result[0],
                        fitness=self.result[1],
                        reward=self.best_reward,
                        history=self.log_history if self.log_history else checkpoint_file,
                        duration=(datetime.now() - time_start).total_seconds(),
                        start=time_start,
                        parameters=parameters,
                        )

        return None

    def get_agent_parameters(self, parameters, allow_mutable=True):
        return parameters

    @property
    def agent_file(self):
        return self.get_agent_file(self.log_file)

    @property
    def checkpoint_file(self):
        return self.get_checkpoint_file(self.log_file)

    def dump(self, parameters, file_name=None, verbose=False):
        if not self.is_root:
            return None

        file_name = self.checkpoint_file if file_name is None else file_name
        if file_name is None:
            return

        if verbose:
            self.persist(f'saving agent repr to: `{file_name}`.')

        import yaml
        from yaml.representer import RepresenterError

        head, tail = os.path.split(file_name)
        os.makedirs(head, exist_ok=True)
        agent_repr = self.get_agent_repr(parameters)

        if self.dump_models:
            for model in agent_repr.get('unfreeze', []):
                model_file = self.get_model_file(model, file_name=file_name)
                if model_file is None:
                    continue

                with open(model_file, 'w') as f:
                    try:
                        if model not in agent_repr:
                            critic_model = agent_repr['critic']
                            for attr in model.split('.'):
                                critic_model = critic_model[attr]

                            dump_model = critic_model
                        else:
                            dump_model = agent_repr[model]

                        yaml.safe_dump(dump_model, stream=f, default_flow_style=None, sort_keys=False)

                    except RepresenterError as ex:
                        self.persist(
                            f"WARNING: yaml-representation might by corrupted: {str(ex)}, saving without safe_dump.")
                        yaml.dump(agent_repr[model], stream=f, default_flow_style=None, sort_keys=False)

        else:
            with open(file_name, 'w') as f:
                try:
                    yaml.safe_dump(agent_repr, stream=f, default_flow_style=None, sort_keys=False)

                except RepresenterError as ex:
                    self.persist(f"WARNING: yaml-representation might by corrupted: {str(ex)}, saving without safe_dump.")
                    yaml.dump(agent_repr, stream=f, default_flow_style=None, sort_keys=False)

        return file_name

    def get_model_file(self, model_name, file_name=None):
        file_name = self.checkpoint_file if file_name is None else file_name
        if file_name is None:
            return

        prefix = '.'.join(file_name.split('.')[:-1])
        postfix = '.' + file_name.split('.')[-1]
        return prefix + '-' + model_name + postfix

    def get_agent_repr(self, parameters):
        # with World.make(self.world if not isinstance(self.world, list) else self.world[0]) as w:
        #     w.set_parameters(parameters)
        #     agent = w.agent
        w = self.world
        if not isinstance(self.world, World):
            # TODO: use World.make(...) to generate a `WorldList` instance ...
            w = World.make(self.world if not isinstance(self.world, list) else self.world[0])

        parameters = self.get_agent_parameters(parameters, allow_mutable=False)
        if not isinstance(parameters, tuple):  # relevant for NEAT
            parameters = parameters.flatten()

        w.set_parameters(parameters)
        agent = w.agent
        return agent.to_dict()

    @classmethod
    def load_log(cls, filename, to_pandas=False) -> list:
        with open(filename, 'r') as f:
            h = [line.strip() for line in f.readlines()]

        runs, run = [], {}
        in_head, in_body, in_tail = True, False, False

        for i, l in enumerate(h):
            if not l:
                continue

            if l.startswith('ES: initializing'):
                in_head, in_body, in_tail = True, False, False

                runs.append({})
                run = runs[-1]

            if not l.startswith('Iteration'):
                if in_head:
                    in_body, in_tail = False, False

                    head = run.get('head', [])
                    head.append(l)
                    run['head'] = head

                else:
                    in_head, in_body, in_tail = False, False, True

                    tail = run.get('tail', [])
                    tail.append(l)
                    run['tail'] = tail

            else:
                in_head, in_body, in_tail = False, True, False

                data = run.get('data', [])

                d, *duration = l.split(', Duration: ')

                diversity_stat = ()
                if "Diversity: " in d:  # if due to legacy
                    d, *diversity_stat = d.split(', Diversity: ')

                if "Avg(Std)-Reward " in d:
                    # legacy branch
                    d, *fitness_stat = d.split(', Avg(Std)-Reward ')
                else:
                    d, *fitness_stat = d.split(', Avg(Std)-Reward: ')

                d, *fitness_current = d.split(', Current Reward: ')
                d, *fitness_best = d.split(' -> Best Reward: ')
                d, *iteration = d.split('Iteration')

                # prepare data
                d = {}

                # iteration
                step, max_steps = [int(li) for li in iteration[0].split('/')]
                d['step'] = step
                d['max_steps'] = max_steps

                # fittest
                d['best'] = float(fitness_best[0])

                # fitness
                d['cost'] = float(fitness_current[0])

                # stats
                d['mean'], d['std'] = fitness_stat[0].replace(')', '').split('(')
                d['mean'], d['std'] = float(d['mean']), float(d['std'])

                # diversity
                if diversity_stat:
                    d['diversity_mean'], d['diversity_std'] = diversity_stat[0].replace(')', '').split('(')
                    d['diversity_mean'], d['diversity_std'] = float(d['diversity_mean']), float(d['diversity_std'])

                # duration
                d['duration'] = duration[0]
                if ' ' in d['duration']:
                    d['duration'], *duration_units = duration[0].split(' ')
                    d['duration_units'] = duration_units[0]

                d['duration'] = float(d['duration'])

                # append data
                data.append(d)
                run['data'] = data

        # clean up
        for run in runs:
            if 'head' not in run:
                run['head'] = []

            if 'data' not in run:
                run['data'] = {'step': [],
                               'max_steps': [],
                               'best': [],
                               'cost': [],
                               'mean': [],
                               'std': [],
                               'duration': [],
                               }

            if 'tail' not in run:
                run['tail'] = []

        # add meta data
        for run in runs:
            head = run["head"]
            try:
                es_util = [line.split("`")[-2].strip() for line in head if "loading es-util solver" in line][0]
            except IndexError:
                es_util = "unknown"

            try:
                solver_options = []
                in_sopts = False
                for line in head:
                    in_sopts |= "solver options" in line
                    if "ES:" in line:
                        in_sopts = False

                    if in_sopts:
                        if "{" not in line:
                            solver_options[-1] = solver_options[-1] + line
                        else:
                            solver_options.append(line)

                opts = [":".join(line.split(":")[1:]).strip() for line in solver_options][0]

            except IndexError:
                opts = {}

            try:
                population_size = \
                [ast.literal_eval(line.split(":")[-1].strip()) for line in head if "population-size" in line][0]
            except IndexError:
                population_size = -1

            try:
                num_parameters = [ast.literal_eval(line.split(":")[-1].strip().split(' ')[0]) for line in head if
                                  "num. parameters" in line][0]
            except IndexError:
                num_parameters = -1

            run["head"] = {"es_util": es_util, "opts": opts, "population_size": population_size,
                           "num_parameters": num_parameters}

        # convert to pandas if requested
        if to_pandas:
            import pandas as pd
            for run in runs:
                run['data'] = pd.DataFrame(run['data'])

        return runs

    @classmethod
    def get_default_opts(cls):
        return {}

    @classmethod
    def load_opts(cls, opts, *args, comm=None, **opt_kwargs):
        import os
        import yaml
        import json

        if comm is None:
            comm = cls.comm

        if comm.rank == 0:
            if opts is None:
                opts = cls.get_default_opts()

            elif isinstance(opts, str):
                if os.path.isfile(opts):
                    with open(opts, 'r') as f:
                        if opts.endswith('.yml') or opts.endswith('.yaml'):
                            opts = yaml.safe_load(f)

                        elif opts.endswith('.json'):
                            opts = json.load(f)

                        else:
                            raise NotImplementedError(f"Unknown `opts` file-format `{opts}`.")

                else:
                    opts = json.loads(opts)

                    if isinstance(opts, tuple):
                        assert len(opts) == 2, "Require (loadable-opts, {optimizer-kwargs}-dict)-pair."
                        opts, opt_kwargs = opts
                        return cls.load_opts(opts, *args, comm=comm, **opt_kwargs)

            # extract signature kwargs from opts dictionary and treat separately
            import inspect

            def get_keywords(base_cls):
                super_signature = inspect.signature(base_cls)
                keys = [k for k in super_signature.parameters.keys()]
                if "kwargs" in keys:
                    keys.remove("kwargs")
                    for super_super_cls in base_cls.__bases__:
                        keys.extend(get_keywords(super_super_cls))

                return keys

            cls_kws = get_keywords(cls)
            opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in cls_kws}
            loaded_opt_kwargs = {opt: opts.pop(opt)
                                 for opt in cls_kws
                                 if opt in opts and opt not in opt_kwargs}

            opt_kwargs = {**opt_kwargs, **loaded_opt_kwargs}

        opts = comm.bcast(opts, root=0)
        opt_kwargs = comm.bcast(opt_kwargs, root=0)

        return opts, opt_kwargs

    @classmethod
    def load_checkpoint(cls, log_file, runs=-1, gens=-1):

        single_ckpt = not hasattr(runs, "__iter__") and not hasattr(gens, "__iter__")
        if single_ckpt:
            runs, gens = [runs], [gens]

        x0 = []
        ckpt_file = cls.get_checkpoint_file(log_file)
        with h5py.File(ckpt_file, 'r') as h5:
            for r, g in zip(runs, gens):
                run_key = str(sorted([int(k) for k in h5.keys()])[int(r)])
                run_data = h5[run_key]

                x_data = run_data['x']
                gen_key = str(sorted([int(k) for k in x_data.keys()])[int(g)])
                gen_x = x_data[gen_key][()]

                x0.append(gen_x)

        if single_ckpt:
            return x0[0]

        return x0

    @classmethod
    def load_multilog(cls, logfile_dataframe, select_run=-1, scale_by=None, sort_by=None):
        data = {"num_runs": [],
                "num_done": [],
                "steps": [],
                "best": [],
                "fitness": [],
                "mean": [],
                "std": [],
                "solver": [],
                "population_size": [],
                "num_parameters": [],
                }

        time_series = "best", "fitness", "mean", "std"

        max_steps = 0
        for logfile in logfile_dataframe["logfile"]:
            runs = cls.load_log(logfile)

            data["num_runs"].append(len(runs))
            data["num_done"].append(len([r for r in runs if r['tail'] != []]))

            head_data = runs[select_run]['head']
            data["solver"].append(head_data["es_util"])
            data["population_size"].append(head_data["population_size"])
            data["num_parameters"].append(head_data["num_parameters"])

            run_data = runs[select_run]['data']
            data["steps"].append(run_data["step"])
            data["best"].append(run_data["best"])
            data["fitness"].append(run_data["cost"])
            data["mean"].append(run_data["mean"])
            data["std"].append(run_data["std"])

            max_steps = max([max_steps, len(run_data["step"])])

        for i in range(len(logfile_dataframe)):
            for ts in time_series:
                dts = data[ts][i]
                if len(dts) < max_steps:
                    zeros_to_add = max_steps - len(dts)
                    data[ts][i] = np.concatenate((dts, np.zeros(zeros_to_add)))

            if len(data["steps"][i]) < max_steps:
                start_idx = len(data["steps"][i])
                data["steps"][i] = np.concatenate((data["steps"][i], np.arange(start_idx, max_steps)))

        data["steps"] = np.stack(data["steps"])
        data["best"] = np.stack(data["best"])
        data["mean"] = np.stack(data["mean"])
        data["std"] = np.stack(data["std"])
        data["fitness"] = np.stack(data["fitness"])

        df = copy.copy(logfile_dataframe[[k for k in logfile_dataframe.keys() if k not in ("logfile", "checkpoint")]])
        df["es-util"] = data["solver"]
        df["pop-size"] = data["population_size"]
        df["num-params"] = data["num_parameters"]
        df["max-fitness"] = data["best"].max(axis=1)

        if scale_by:
            df["scale_by"] = scale_by
            for val, grid_XY in df.groupby(scale_by):
                scale = max(grid_XY["max-fitness"])
                df.loc[grid_XY.index, "scale"] = scale
                data["fitness"][grid_XY.index] /= scale
                data["best"][grid_XY.index] /= scale
                data["mean"][grid_XY.index] /= scale
                data["std"][grid_XY.index] /= scale

        if sort_by:
            df = df.sort_values(sort_by, ascending=False)
            data["fitness"] = data["fitness"][df.index]
            data["best"] = data["best"][df.index]
            data["mean"] = data["mean"][df.index]
            data["std"] = data["std"][df.index]
            df = df.reset_index(drop=True)

        return data, df

    @classmethod
    def load_multirun(cls, df, time_series=("best", "fitness", "mean", "std"), outliers=()):
        max_steps = 0
        data = {"num_runs": [],
                "num_done": [],
                "steps": [],
                "best": [],
                "fitness": [],
                "mean": [],
                "std": [],
                "solver": [],
                "population_size": [],
                "num_parameters": [],
                }

        for i, logfile in zip(df.index, df["logfile"]):
            runs = cls.load_log(logfile, to_pandas=True)

            data["num_runs"].append(len(runs))
            data["num_done"].append(len([r for r in runs if r['tail'] != []]))

            run_data = {"solver": [], "population_size": [], "num_parameters": [],
                        "steps": [], "best": [], "fitness": [], "mean": [], "std": []}

            potential_outliers = []
            for outlier in outliers:
                if len(df.iloc[[i]].query(" and ".join(
                        [(f"{k} == {v}" if not isinstance(v, str) else f"{k} == '{v}'") for k, v in outlier.items() if
                         k != "run"]))):
                    print("potential outlier:", logfile)
                    potential_outliers.append(outlier)

            for r_i, run in enumerate(runs):
                outlier = None
                outlier_detected = False
                for outlier in potential_outliers:
                    outlier_detected |= (r_i == outlier["run"])
                    if outlier_detected:
                        break

                if outlier_detected:
                    print(f"detected outlier {outlier}")
                    data["num_runs"][-1] -= 1
                    continue

                head_data = run['head']
                run_data["solver"].append(head_data["es_util"])
                run_data["population_size"].append(head_data["population_size"])
                run_data["num_parameters"].append(head_data["num_parameters"])

                run_data["steps"].append(run["data"]["step"])
                run_data["best"].append(run["data"]["best"])
                run_data["fitness"].append(run["data"]["cost"])
                run_data["mean"].append(run["data"]["mean"])
                run_data["std"].append(run["data"]["std"])

            max_steps = max([max_steps, max([len(s) for s in run_data["steps"]])])
            [data[k].append(v) for k, v in run_data.items()]

        for i in range(len(df)):
            for j in range(data["num_runs"][i]):
                num_steps = len(data["steps"][i][j])
                if num_steps < max_steps:
                    fill_nan = [np.nan] * (max_steps - num_steps)
                    data["steps"][i][j] = np.concatenate((data["steps"][i][j], fill_nan))
                    for ts in time_series:
                        data[ts][i][j] = np.concatenate((data[ts][i][j], fill_nan))

            for ts in time_series:
                data[ts][i] = np.stack(data[ts][i])
            data["steps"][i] = np.stack(data["steps"][i])

        df = copy.copy(df[[k for k in df.keys() if k not in ("logfile", "checkpoint")]])
        for k, v in data.items():
            df[k] = v

        df["max-fitness"] = [np.max(best) for best in df["best"]]
        return df, list(data.keys()) + ["max-fitness"]


def setup_optimizer(world_repr, es_util, new_model=False, comm=None, opts=None, es_method=None):
    kwargs = {}
    if 'NEAT' == es_util:
        from .neat_method import NEATMethod as es_method

    elif 'HyperNEAT' == es_util:
        # assert es_util != 'HyperNEAT', "`HyperNEAT` not implemented."
        from .hyper_neat_method import HyperNEATMethod as es_method

    elif 'BoostES' == es_util:
        from .boost_es_method import BoostES as es_method

    else:
        if es_method is None:
            es_method: type = EvolutionaryStrategy

        kwargs['es_util'] = es_util

    input_size, action_size = None, None
    if comm is None:
        comm = es_method.comm

    try:
        x0 = None
        if comm.rank == 0:
            with World.make(world_repr) as w:
                input_size = w.agent.observation_size
                action_size = w.agent.action_size
                if not new_model:
                    x0 = w.get_parameters()

    except (FileNotFoundError, TypeError):
        from warnings import warn
        warn("Could not load `x0` from `world_repr`.")
        x0 = None

    opts, opt_kwargs = es_method.load_opts(opts, input_size=input_size, comm=comm, action_size=action_size, **kwargs)
    return es_method, x0, opts, opt_kwargs
