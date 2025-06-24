import neat
from neat.six_util import iterkeys
import pickle
from mindcraft.io import Repr
from mindcraft.train.es_method import EvolutionaryStrategy
import os
import numpy as np
from ast import literal_eval


def getkeyattr(obj, key, *default):
    """ Retrieves object attribute via key, either via `getattr`, with `obj.get`, or with `obj[key]` in that order.

    :param obj: Arbitrary python object which holds an attribute/property/key-value pair specified via `key`.
    :param key: The string identifier of the requested attribute/property/key-value pair.
    :param default: Default values for `getattr` function or `obj.get` method.
    :return: The object's attribute corresponding to the `key` attribute-name.

    (c) B. Hartl 2021
    """
    try:
        return getattr(obj, key, *default)

    except AttributeError:
        if len(default) > 0:
            return obj.get(key, *default)

        return obj[key]


class Population(neat.Population):
    """ A `neat.Population` wrapper for the `mindcraft` framework

    (c) B. Hartl 2021
    """
    def __init__(self, config, initial_state=None, es_solver: (EvolutionaryStrategy, None) = None):

        self.log = None
        self._es_solver = None
        self.es_solver = es_solver

        self.currently_best_genome = None
        self.current_fitness_list = None

        if initial_state is not None and not isinstance(initial_state, (tuple, list, neat.Population)):
            initial_state = self.load(initial_state)

        if isinstance(initial_state, neat.Population):
            initial_state = initial_state.population, initial_state.species, initial_state.generation + 1

        neat.Population.__init__(self, config, initial_state=initial_state)

    @property
    def es_solver(self):
        return self._es_solver

    @es_solver.setter
    def es_solver(self, value):
        self._es_solver = value

        if self._es_solver is None:
            self.log = lambda *args, **kwargs: None

        else:
            try:
                assert hasattr(self._es_solver, 'persist')
                self.log = self._es_solver.persist

            except AssertionError:
                self.log = print

    def load(self, checkpoint: (str, int) = -1):
        """ load population via checkpoint file or generation index of checkpoint.

        If `checkpoint` is used as file-name (without prefix) or as index to a generation or to a chronologically
        ordered checkpoint, the path to the checkpoint files needs to be specified via the `es_solver` property.

        :param checkpoint: path to checkpoint file, checkpoint file-name, generation index or chronological index
                           of checkpoint file.
        """

        if isinstance(checkpoint, str):
            try:
                checkpoint_int = literal_eval(checkpoint)
                checkpoint = checkpoint_int
            except SyntaxError:
                pass

        if isinstance(checkpoint, int):
            checkpoint = self.find_checkpoint(index=checkpoint, prefix=self.checkpoint_prefix)
            checkpoint = str(checkpoint)

        if not os.path.isfile(checkpoint):
            checkpoint = "{}{}".format(self.checkpoint_prefix, checkpoint)

        self.log('restoring from checkpoint `{}`'.format(checkpoint))
        population = self.restore_checkpoint(checkpoint)

        return population

    def find_checkpoint(self, index, prefix):
        """ find checkpoint file where

        - index is a certain generation, or where
        - index can be used to refer to certain checkpoints (e.g. to the most recent one via `index=-1`

        :param index: identifier of either a specific generation's checkpoint file or as list index of
                      all checkpoint files.
        :param prefix: prefix or file-scheme where checkpoint files are located
        """

        from os import listdir
        import os

        path = os.path.dirname(prefix)  # path to checkpoints
        file = prefix[len(path) + 1:]   # filename (scheme) of prefix

        checkpoints = [int(f[len(file):]) for f in listdir(path) if f.startswith(file)]  # extract literals
        checkpoints = sorted(checkpoints)  # sort chronologically

        if index in checkpoints:  # return specific checkpoint index
            return index

        return checkpoints[index]  # interpret index referring to a specific checkpoints

    @property
    def checkpoint_prefix(self):
        """file prefix (str) to neat-checkpoint files is defined by log-file + "checkpoint-"."""
        checkpoint_prefix = self.es_solver.log_file
        checkpoint_prefix = '.'.join(checkpoint_prefix.split('.')[:-1])  # remove logfile ending (usually .log)

        checkpoint_prefix += ('-' * (not checkpoint_prefix.endswith('/'))) + 'checkpoint-'
        return checkpoint_prefix

    @classmethod
    def restore_checkpoint(cls, filename):
        """ Resumes the simulation from a previous saved point.

        Adapted from neat.checkpoint.Checkpointer under the BSD 3-Clause Licence (Sept. 2021)
        """
        import gzip
        import random

        with gzip.open(filename) as f:
            generation, config, population, species_set, rnd_state = pickle.load(f)
            random.setstate(rnd_state)
            return cls(config, (population, species_set, generation))

    def ask(self):
        """ get solutions from ES solver

        Adapted from neat.population.Population run under the BSD 3-Clause Licence (Sept. 2021)
        """
        self.reporters.start_generation(self.generation)
        return [(self.population[i], self.config) for i in sorted(self.population.keys())]

    def update_reproduction_counter(self):
        keys = self.population.keys()

        try:
            max_key = max(keys)
        except ValueError:
            max_key = self.reproduction.genome_indexer.__reduce__()[1][0]

        while self.reproduction.genome_indexer.__reduce__()[1][0] < max_key:
            next(self.reproduction.genome_indexer)

        return keys

    def tell(self, reward_table_result):
        """ submit reward table results to ES solver

        Adapted from neat.population.Population run under the BSD 3-Clause Licence (Sept. 2021)
        """
        keys = self.update_reproduction_counter()
        f2k = sorted(keys)

        best = None
        for i, fitness in enumerate(reward_table_result):
            g = self.population[f2k[i]]
            g.fitness = fitness

            if best is None or g.fitness > best.fitness:
                best = g

        self.currently_best_genome = best
        self.current_fitness_list = reward_table_result
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if self.best_genome is None or not hasattr(self.best_genome, 'fitness'):
            self.best_genome = best
        elif self.best_genome.fitness is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        # if not self.config.no_fitness_termination:
        #     # End if the fitness threshold is reached.
        #     fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
        #     if fv >= self.config.fitness_threshold:
        #         self.reporters.found_solution(self.config, self.generation, best)
        #         break

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species, self.config.pop_size, self.generation)

        # assure a fixed maximum population size
        # if len(self.population) > self.config.pop_size * 2.:
        #     n_reduce = len(self.population) - self.config.pop_size
        #     self.log(f'# population size {len(self.population)} too large, reduce by {n_reduce} genomes')
        #     self.__sub__(n_reduce)

        if len(self.population) < self.config.pop_size:
            new_population = self.reproduction.create_new(self.config.genome_type,
                                                          self.config.genome_config,
                                                          self.config.pop_size - len(self.population))

            self.__add__(new_population)

            # also add current best genome if self.config.reset_with_best is set
            if self.config.reset_with_best and np.random.rand() < abs(self.config.reset_with_best):
                self.__sub__(1)
                if self.best_genome.key not in self.population:
                    self.population[self.best_genome.key] = self.best_genome
                else:
                    best_variation = self.es_solver.vary_genome(self.best_genome, copy=True)
                    self.population[best_variation.key] = best_variation

                self.log(f'# population size too small, extend by {len(new_population)} new genomes (*)')
            else:
                self.log(f'# population size too small, extend by {len(new_population)} new genomes')

            self.update_reproduction_counter()

        # the number `3` below is rather arbitrary, maybe this should be a parameter?
        n_max = 4 * self.config.pop_size
        if len(self.population) > n_max:
            n = int(len(self.population) - self.config.pop_size * 3.5)
            self.log(f"# reproduction exceeds ({n_max//self.config.pop_size} x pop-size), removing {n} individuals.")
            self.__sub__(n)
            self.update_reproduction_counter()

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()
            self.log('# complete extinction of species detected')

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)

                if self.config.reset_with_best:
                    self.__sub__(1)
                    self.population[self.best_genome.key] = self.best_genome
                    self.log('# resetting population (*)')

                else:
                    self.log('# resetting population')

            else:

                self.log('# raised CompleteExtinctionException()')
                raise neat.population.CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)
        self.reporters.end_generation(self.config, self.population, self.species)
        self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)
        #
        return self.best_genome

    def __add__(self, other):
        """"""
        keys = {k for k in self.population.keys()}

        try:
            max_keys = max(keys)
        except ValueError:
            max_keys = self.reproduction.genome_indexer.__reduce__()[1][0]

        other_population = other.population if isinstance(other, Population) else other

        for genome_id, genome in other_population.items():
            if genome_id in keys or genome_id < max_keys:
                genome.key = max_keys + 1
                keys.add(genome.key)
                max_keys = genome.key

            self.population[genome.key] = genome

        return self.population

    def del_individual(self, key):
        self.population.pop(key)

        for species_key, species in self.species.species.items():
            if key in species.members:
                species.members.pop(key)

    def __iadd__(self, other):
        self.__add__(other)
        return self

    def __sub__(self, other: int):
        """ delete a number of `other` of the worst genomes form the population """
        to_del = [k for k in sorted(self.population.keys()) if self.population[k].fitness is None]

        for o in range(other):
            # delete preferably non-evaluated genomes
            if len(to_del) > 0:
                self.del_individual(key=to_del.pop())
                continue

            # delete worst performing genomes iteratively
            worst_fitness = np.inf
            worst_key = None
            for key, genome in self.population.items():
                if genome.fitness < worst_fitness:
                    worst_key = key
                    worst_fitness = genome.fitness

            self.del_individual(key=worst_key)

        return self.population

    def __isub__(self, other):
        self.__sub__(other)
        return self


class ConfigWrapper(Repr, neat.Config):
    """ A `neat.Config` wrapper for the `mindcraft` framework, utilizing it as `Repr` object

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('opts', 'pop_size',
                   # 'genome_type', 'reproduction_type', 'species_set_type', 'stagnation_type',
                   )

    ALLOWED_CONNECTIVITY = neat.genome.DefaultGenomeConfig.allowed_connectivity

    # NEAT DEFAULT
    DEFAULT_CONFIG = dict(
        neat_config=dict(
            fitness_criterion='max',
            reset_on_extinction=True,
            reset_with_best=0.1,
            no_fitness_termination=True,
            fitness_threshold=0,
        ),
        genome_config=dict(
            # network_parameters
            feed_forward=True,
            initial_connection=['partial', 0.1],  # Create 10% partially connected network
            num_hidden=0,
            num_inputs=2,
            num_outputs=1,
            # node_activation_options
            activation_default='sigmoid',
            activation_options='sigmoid',
            activation_mutate_rate=0.,
            # node_aggregation_options
            aggregation_default='sum',
            aggregation_mutate_rate=0.0,
            aggregation_options='sum',
            # node_topology_options
            node_add_prob=0.025,
            node_delete_prob=0.03,
            # node_bias_options
            bias_init_mean=0.0,
            bias_init_stdev=1.0,
            bias_max_value=30.0,
            bias_min_value=-30.0,
            bias_mutate_power=0.5,
            bias_mutate_rate=0.1,
            bias_replace_rate=0.05,
            # node_response_options
            response_init_mean=1.0,
            response_init_stdev=0.0,
            response_max_value=30.0,
            response_min_value=-30.0,
            response_mutate_power=0.0,
            response_mutate_rate=0.0,
            response_replace_rate=0.0,
            # connection_topology_options
            conn_add_prob=0.07,
            conn_delete_prob=0.075,
            enabled_default=True,
            enabled_mutate_rate=0.,
            enabled_rate_to_false_add=0.,
            enabled_rate_to_true_add=0.,
            # connection_weight_options
            weight_init_mean=0.0,
            weight_init_stdev=1.0,
            weight_max_value=30,
            weight_min_value=-30,
            weight_mutate_power=0.5,
            weight_mutate_rate=0.1,
            weight_replace_rate=0.05,
            # genome_compatibility_options
            compatibility_disjoint_coefficient=1.0,
            compatibility_weight_coefficient=0.5,
            # structure
            single_structural_mutation=False,
            structural_mutation_surer="default",
        ),
        species_set_config=dict(
            compatibility_threshold=3.0,
        ),
        stagnation_config=dict(
            species_fitness_func='max',
            max_stagnation=20,
        ),
        reproduction_config=dict(
            elitism=2,
            survival_threshold=0.2,
            min_species_size=2,
        )
    )

    DEFAULT_CONFIG_MAPPING = {
        'neat_config': 'NEAT',
        'genome_config': 'Genome',
        'species_set_config': 'SpeciesSet',
        'stagnation_config': 'StagnationConfig',
        'reproduction_config': 'ReproductionConfig'
    }

    def __init__(self,
                 pop_size,
                 opts=None,
                 ):

        Repr.__init__(self, repr_fields=self.REPR_FIELDS)

        self.opts = opts if opts is not None else dict()
        for name in self.DEFAULT_CONFIG_MAPPING.keys():
            default = self.DEFAULT_CONFIG.get(name, {})
            opt_name = self.DEFAULT_CONFIG_MAPPING[name]
            config = self.opts.get(name, self.opts.get(opt_name, default))

            for k_i, v_i in default.items():
                if k_i not in config:
                    config[k_i] = v_i  # set default value

            self.set_opts(name, config)

        self.opts[self.DEFAULT_CONFIG_MAPPING['neat_config']]['pop_size'] = pop_size

        self.__init_config()

    def set_opts(self, name, values):
        _property = self.opts.pop(name, dict())

        for k, v in values.items():
            _property[k] = v

        try:
            name = self.DEFAULT_CONFIG_MAPPING[name]
        except KeyError:
            pass

        self.opts[name] = _property

    def __init_config(self, genome_type=None, reproduction_type=None, species_set_type=None, stagnation_type=None):
        """ Adapted from neat-python's neat.Config.__init__ under the BSD 3-Clause Licence (Sept. 2021) """

        # Check that the provided types have the required methods.
        if genome_type is None:
            genome_type = neat.DefaultGenome
        assert hasattr(genome_type, 'parse_config')

        if reproduction_type is None:
            reproduction_type = neat.DefaultReproduction
        assert hasattr(reproduction_type, 'parse_config')

        if species_set_type is None:
            species_set_type = neat.DefaultSpeciesSet
        assert hasattr(species_set_type, 'parse_config')

        if stagnation_type is None:
            stagnation_type = neat.DefaultStagnation
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        # NEAT configuration
        if 'NEAT' not in self.opts:
            raise RuntimeError("'NEAT' section not found in NEAT configuration.")

        param_list_names = []
        param_dict = self.opts['NEAT']
        for p, v in param_dict.items():
            setattr(self, p, v)
            param_list_names.append(p)

        unknown_list = [x for x in iterkeys(param_dict) if not x in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise neat.config.UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise neat.config.UnknownConfigItemError(
                "Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        def get_naming(cls):
            name = cls.__name__
            if name not in self.opts and name.startswith('Default'):
                return name[len('Default'):]
            return name

        def neat_params(dict_repr):
            parsed_dict = {}
            for k, v in dict_repr.items():
                if not isinstance(v, str) and hasattr(v, '__iter__'):
                    v = " ".join([str(vi) for vi in v])

                parsed_dict[k] = str(v)

            return parsed_dict

        genome_dict = self.opts.get(get_naming(genome_type), dict())
        self.genome_config = genome_type.parse_config(neat_params(genome_dict))

        species_set_dict = self.opts.get(get_naming(species_set_type), dict())
        self.species_set_config = species_set_type.parse_config(neat_params(species_set_dict))

        stagnation_dict = self.opts.get(get_naming(stagnation_type), dict())
        self.stagnation_config = stagnation_type.parse_config(neat_params(stagnation_dict))

        reproduction_dict = self.opts.get(get_naming(reproduction_type), dict())
        self.reproduction_config = reproduction_type.parse_config(neat_params(reproduction_dict))

    def save(self, filename):
        """ save neat.Config file to file specified via `solver.opts.export_path` settings used instead"""
        self.to_yml(filename)

    # def to_dict(self):
    #     from mindcraft.torch.wrapper.graph.wiring import
    #     dict_repr = Repr.to_dict(self)
    #     opts = dict_repr.pop('opts')
    #     for k, v in self.DEFAULT_CONFIG_MAPPING.items():
    #         dict_repr[k] = opts.get(v, dict())
    #
    #     dict_repr["genome_config"]["input_keys"] = self.genome_config.input_keys
    #     dict_repr["genome_config"]["output_keys"] = self.genome_config.output_keys
    #
    #     dict_repr.pop("cls")
    #     dict_repr.pop("locate")
    #     dict_repr.pop("pop_size")
    #     return dict_repr


class NEAT(Repr):
    """ NEAT wrapper of the `mindcraft` framework

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('popsize', )

    def __init__(self,
                 pop_size,
                 opts,
                 x0=None,
                 init_population=True,
                 es_solver=None,
                 **kwargs
                 ):
        """

        :param pop_size: Population size.
        :param opts: Solver options in dict format.
        :param x0: Default-Genome object, or path to pkl-file, or dict-like parse-able genome, which is used as initial
                   value in the evolutionary population.
        :param init_population: (i) either boolean, controlling whether a random population should be initialized, or
                                (ii) path to a checkpoint file,
                                (iii) string identifier 'by_x0' to trigger flushing the initial population with varied
                                      versions of the `x0` genome, or
                                (iv) a (population, species, generation) tuple accepted by the neat.population.Population
                                     constructor.
        :param es_solver:
        :param kwargs:
        """

        Repr.__init__(self, repr_fields=self.REPR_FIELDS)

        self.net = None

        # init neat
        self.config = ConfigWrapper(pop_size=pop_size, opts=opts)

        # init population
        self.population = None
        self._init_population = None
        self._es_solver = None
        self.es_solver = es_solver

        flush_x0 = isinstance(init_population, str) and init_population == 'by_x0'
        if flush_x0:
            init_population = True

        try:
            self.init_population = init_population
        except IndexError:  # if checkpoint does not exist
            self.init_population = True

        self.flush_x0 = flush_x0
        self._x0 = None
        try:
            self.x0 = x0
        except IndexError:
            pass

    @property
    def es_solver(self):
        return self._es_solver

    @es_solver.setter
    def es_solver(self, value):
        self._es_solver = value

        if self._es_solver is None:
            self.log = lambda *args, **kwargs: None

        else:
            try:
                assert hasattr(self._es_solver, 'persist')
                self.log = self._es_solver.persist

            except AssertionError:
                self.log = print

    @property
    def init_population(self):
        return self._init_population

    @init_population.setter
    def init_population(self, value):
        if value:
            self.population = Population(config=self.config,
                                         initial_state=value if not isinstance(value, bool) else None,
                                         es_solver=self._es_solver)

            pop_size = self.config.pop_size
            loaded_size = len(self.population.population)
            if pop_size != loaded_size:
                self.log(f'init_population: change loaded pop-size from {loaded_size} '
                         f'to config-value of {pop_size}.')

                if loaded_size < pop_size:  # need to add genomes
                    self.config.pop_size = self.config.pop_size - len(self.population.population)
                    filler_population = Population(config=self.config,
                                                   initial_state=None,
                                                   es_solver=self._es_solver)
                    self.population += filler_population
                    self.config.pop_size = pop_size

                elif loaded_size > pop_size:  # delete genomes
                    self.population -= (loaded_size - pop_size)

                self.population.species.speciate(self.config, self.population.population, self.population.generation)

        self._init_population = value

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value: (neat.DefaultGenome, dict)):
        """ Add x0 to the population of possible solutions.

        If self.flush_x0 is set, the entire population will be initialized with varied versions of x0,
        x0 itself being the first element of the population. Otherwise, x0 will replace the population
        element with the least fitness, or any which has fitness == None.

        Eventually, a 'speciation' call is launched.
        """
        if value is not None:

            if not isinstance(value, neat.DefaultGenome):
                value = self.parse_genome(value)

            if self.flush_x0:
                self.log('flushing population with `x0`')
                keys = sorted(list(self.population.population.keys()))
                for i, key in enumerate(keys):
                    if i == 0:
                        self.population.population[value.key] = value
                    else:
                        varied_g = self.vary_genome(value, copy=True)
                        varied_g.fitness = None
                        self.population.population[varied_g.key] = varied_g

                    self.population.population.pop(key)

            else:
                self.log('including `x0` in population')
                self.population -= 1
                self.population.population[value.key] = value

            self.population.species.speciate(self.config, self.population.population, self.population.generation)

        self._x0 = value

    def parse_genome(self, value):
        """ parse a neat-nome from a dict-like- or DefaultGenome value and return a
            proper Genome of self.config.genome_type (in any-case a copy of value) """

        if isinstance(value, str):
            value = self.load_genome(filename=value)

        if not isinstance(value, neat.DefaultGenome):
            max_keys = max([k for k in self.population.population.keys()])
            genome = self.config.genome_type(key=max_keys + 1)  # new, unique key

        else:
            genome = value.__class__(key=value.key)

        gene_config = self.config.genome_config
        for key, attrs in getkeyattr(value, 'nodes').items():
            if key < 0:  # no need for input nodes, covered by config
                continue

            node = gene_config.node_gene_type(key)
            for a in node._gene_attributes:
                v = getkeyattr(attrs, a.name)
                setattr(node, a.name, v)

            genome.nodes[key] = node

        try:
            edges = getkeyattr(value, 'edges')
        except (AttributeError, KeyError, TypeError):
            edges = getkeyattr(value, 'connections')

        try:  # neat connection representation
            for (input_key, output_key), connection in edges.items():
                weight = getkeyattr(connection, 'weight')
                enabled = getkeyattr(connection, 'enabled')
                genome.add_connection(gene_config, input_key, output_key, weight, enabled)

        except TypeError:  # allow graph-adjacency representation
            for input_key, edge_u in edges.items():
                for output_key, connection in edge_u.items():
                    weight = getkeyattr(connection, 'weight')
                    enabled = getkeyattr(connection, 'enabled')
                    genome.add_connection(gene_config, input_key, output_key, weight, enabled)

        return genome

    def vary_genome(self, g, attrs=('bias', 'response', 'weight'), copy=False):
        """ vary all '*_init_stdev attributes of the genome 'g' which are defined in the 'genome_config'
            section of the opts and specified via the '*attrs' opts.
            Return a copy if the 'copy' flag is set. """
        if copy:
            g = self.parse_genome(g)
            g.key = max([k for k in self.population.population.keys()]) + 1

        for a in attrs:
            std = getattr(self.config.genome_config, f'{a}_init_stdev')

            if any(n == a for n in ('bias', 'response')):
                var = np.random.randn(len(g.nodes)) * std
                [setattr(node, a, getattr(node, a) + delta_v)
                 for (_, node), delta_v in zip(g.nodes.items(), var)]

            elif any(n == a for n in ('weight', )):
                var = np.random.randn(len(g.connections)) * std
                [setattr(conn, a, getattr(conn, a) + delta_v)
                 for (_, conn), delta_v in zip(g.connections.items(), var)]

            else:
                import warnings
                warnings.warn(f"Unknown attribute `{a}` in vary_genome.")

        return g

    def dump_genome(self, file_name=None, verbose=True):
        """ save best genome to specified 'self.genome_filename' in pkl format """

        file_name = self.get_genome_filename(file_name)

        if verbose:
            self.log('saving genome to `{}`'.format(file_name))

        dir_name = os.path.dirname(file_name)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(self.population.best_genome, f)

        return file_name

    def load_genome(self, filename=None):
        """ load pickled genome from specified 'self.genome_filename' """

        filename = self.get_genome_filename(filename)

        self.log('loading genome from `{}`'.format(filename))
        with open(filename, 'rb') as f:
            genome = pickle.load(f)

        self.population.best_genome = genome
        return genome

    def get_genome_filename(self, filename=None):
        """ standard filename for genome-io, based on self._es_solver's logfile """

        if filename is None:
            filename = self._es_solver.log_file

        if not filename.endswith('.pkl'):
            is_dir = filename.endswith('/') or filename.endswith('\\')
            if is_dir:
                filename = os.path.join(filename, 'genome.pkl')

            else:
                filename = os.path.splitext(filename)[0]  # remove extension
                filename += '-genome.pkl'

        filename = filename.replace('//', '/').replace('\\', '/')  # enforce unix format
        return filename

    def ask(self):
        """ get solutions from ES solver"""
        return self.population.ask()

    def tell(self, reward_table_result):
        """ submit reward table results to ES solver"""
        return self.population.tell(reward_table_result)

    def result(self):
        """ return tuple of (best params so far, historically best reward, curr reward, sigma) """

        best_genome, best_fitness = None, None
        if self.population.best_genome is not None:
            best_genome = self.population.best_genome
            best_fitness = best_genome.fitness

        current_reward, current_sigma = None, None
        if self.population.currently_best_genome is not None:
            current_reward = self.population.currently_best_genome.fitness
            current_sigma = np.std(self.population.current_fitness_list)

        return (best_genome, self.config), best_fitness, current_reward, current_sigma
