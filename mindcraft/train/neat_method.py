import neat
from mindcraft.torch.wrapper.graph import Wiring
from mindcraft.torch.wrapper.graph.wiring import neat_to_digraph, digraph_to_dict
from mindcraft.train import Compose
from mindcraft.train import EvolutionaryStrategy
from types import MappingProxyType

# required for dynamically loading stuff:
from mindcraft import World
import numpy as np


class NEATMethod(EvolutionaryStrategy):
    """ NEAT Optimizer for the `mindcraft` framework

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('population_size', 'sigma_init', 'init_population', 'x0',
                   'opts', 'verbose', 'log_file', 'log_fields', 'log_foos',
                   *Compose.REPR_FIELDS)
    LOG_FIELDS = tuple(f for f in EvolutionaryStrategy.LOG_FIELDS)
    LOG_FOOS = MappingProxyType({'x': "self.dump_neat_genome(*x)",                        # cast genome -> dict
                                 'parameters': "self.dump_neat_population(parameters)"})  # cast [genomes] -> [dicts]

    def __init__(self,
                 population_size,
                 sigma_init=None,
                 init_population='by_x0',
                 x0=None,
                 opts=None,
                 verbose=True,
                 log_file='neat-es.log',
                 log_fields=LOG_FIELDS,
                 log_foos=LOG_FOOS,
                 **kwargs
                 ):
        """ Constructs a NEATMethod object

        :param population_size: Size of NEAT population.
        :sigma_init: (Optional) Numerical value for the initial variation of parameters drawn from a Gaussian dristrubtion.
        :param init_population: (i) either boolean, controlling whether a random population should be initialized, or
                                (ii) path to- / or integer id of- a checkpoint file,
                                (iii) string identifier 'by_x0' to trigger flushing the initial population with varied
                                      versions of the `x0` genome, or
                                (iv) a (population, species, generation) tuple accepted by the
                                     neat.population.Population constructor.
        :param x0: (Optional) Initial parameter value.
        :param opts: (Optional) Dictionary specifying further options of the NEAT algorithm (see WorldOptimizer docs).
        :param verbose: (Optional) Boolean specifying the output-verbosity of the method.
        :param log_file: (Optional) String specifying the file-naming of the method, defaults to `neat-es.log`.
        :param log_fields: (Optional) Log field instructions, see `mindcraft.io.Log` for details.
        :param log_foos: (Optional) Log field functional instructions, see `mindcraft.io.Log` for details.
        """

        kwargs['method'] = 'neat'
        EvolutionaryStrategy.__init__(self,
                                      population_size=population_size,
                                      log_file=log_file,
                                      verbose=verbose,
                                      log_fields=log_fields,
                                      log_foos=log_foos,
                                      **kwargs)

        self.init_population = init_population
        self.sigma_init = sigma_init
        self.opts = opts
        self.x0 = x0
        self.config = None
        self.es = None

    def set_nparams(self, n_params):
        self.n_params = 0

    def evolutionary_strategy(self, **kwargs):
        self.population_size = kwargs.get("population_size", self.population_size)
        self.init_population = kwargs.get("init_population", self.init_population)
        self.sigma_init = kwargs.get("sigma_init", self.sigma_init)
        self.opts = kwargs.get("opts", self.opts)
        self.x0 = kwargs.get("x0", self.x0)

        if self.sigma_init is not None:
            genome_config = self.opts.get('genome_config', dict())
            genome_config['weight_init_stdev'] = self.sigma_init
            genome_config['bias_init_stdev'] = self.sigma_init

        from mindcraft.train.neat_util import NEAT
        self.es = NEAT(pop_size=self.population_size,
                       opts=self.opts,
                       init_population=self.init_population,
                       x0=self.x0,
                       es_solver=self,
                       )

        self.config = self.es.config

        # add checkpointer
        if self.init_population and self.checkpoint_interval:
            self.es.population.add_reporter(neat.Checkpointer(
                     generation_interval=self.checkpoint_interval,
                     filename_prefix=self.es.population.checkpoint_prefix,
                     time_interval_seconds=300,  # default value
                 ))

        return self.es

    @classmethod
    def get_default_opts(cls):
        from mindcraft.train.neat_util import ConfigWrapper
        return ConfigWrapper.DEFAULT_CONFIG

    @classmethod
    def load_opts(cls, opts, input_size, action_size, *args, comm=None, **opt_kwargs):
        if isinstance(opts, dict) and 'config' in opts:
            opt_kwargs = {**opt_kwargs, **opts.get('kwargs', {})}
            opts = opts['config']

        if comm is None:
            comm = cls.comm

        loaded_opts, opt_kwargs = super().load_opts(opts, *args, comm=comm, **opt_kwargs)

        if comm.rank == 0:
            genome_config = loaded_opts.get('genome_config', dict())
            genome_config['num_inputs'] = input_size
            genome_config['num_outputs'] = action_size
            loaded_opts['genome_config'] = genome_config

        loaded_opts = comm.bcast(loaded_opts, root=0)
        return loaded_opts, opt_kwargs

    @classmethod
    def dump_neat_genome(cls, genome, config):
        """ Parse single neat (genome, config)-parameter pair into dict-representation """
        g, (input_size, output_size) = neat_to_digraph(genome, config)
        return digraph_to_dict(g, input_size, output_size)

    @classmethod
    def dump_neat_population(cls, parameters):
        """ Parse list of neat (genome, config)-parameter pairs into list of dict-representations """
        return [cls.dump_neat_genome(*p) for p in parameters]
