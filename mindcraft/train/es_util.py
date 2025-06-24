""" adopted from: https://github.com/hardmaru/estool under the MIT Licence

    also see
        - https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
        - https://blog.otoro.net/2017/11/12/evolving-stable-strategies/

    taken from B. Hartl 2021
"""


import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list, reg='l2'):
    model_param_grid = np.asarray(model_param_list)
    if reg == 'l1':
        return - weight_decay * np.mean(np.abs(model_param_grid), axis=1)

    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def roulette_wheel(f, s=3., eps=1e-12, assume_sorted=False, normalize=False):
    """ Roulette wheel fitness transformation.

    We transform the fitness values f to probabilities p by applying the roulette wheel fitness transformation.
    The roulette wheel fitness transformation is a monotonic transformation that maps the fitness values to
    probabilities. The selection pressure s controls the degree of selection. The higher the selection pressure,
    the more the probabilities are concentrated on the best solutions (s can be positive or negative).

    :param f: torch.Tensor of shape (popsize,), fitness values of the sampled solutions
    :param s: float, selection pressure
    :param eps: float, epsilon to avoid division by zero
    :param assume_sorted: bool, whether to disable sorting of the fitness values and assume that they are already sorted
    :param normalize: bool, whether to normalize the probabilities to sum to 1 (default False, i.e., the sum over
                      the returned scaled probabilities is equal to the sum over the fitness absolute values)
    :return: torch.Tensor of shape (popsize,), indices of the selected solutions
    """
    if not isinstance(f, np.ndarray):
        f = np.ndarray(f)

    indices = np.arange(len(f))
    if not assume_sorted:
        # sort fitness in ascending order
        asc = f.flatten().argsort()
        f = f[asc]

        # restore original order
        indices = np.where(asc[None, :] == indices[:, None])[1]

    total_weight = np.abs(f).sum()

    fs = (f - f.min()) / (f.max() - f.min() + eps)  # normalize fitness values to [0, 1], and sort
    fs = np.exp(s*fs)  # apply selection pressure, s can be positive or negative
    fs = np.cumsum(fs)  # compute cumulative sum

    fs /= fs.sum()
    if not normalize:
        fs *= total_weight

    return fs[indices]


# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES:
    '''CMA-ES wrapper.'''

    def __init__(self, num_params,
                 sigma_init=1.0,
                 popsize=255,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 inopts=None
                 ):
        """Constructs a CMA-ES solver, based on Hannsen's `cma` module.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param popsize: population size.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param inopts: dict-like CMAOptions, forwarded to cma.CMAEvolutionStrategy constructor).
        :param x0: (Optional) either (i) a single or (ii) several initial guesses for a good solution,
                   defaults to None (initialize via `np.zeros(num_parameters)`).
                   In case (i), the population is seeded with x0.
                   In case (ii), the population is seeded with mean(x0, axis=0) and x0 is subsequently injected.
        """

        self.popsize = popsize

        inopts = inopts or {}
        inopts['popsize'] = self.popsize

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.weight_decay = weight_decay
        self.reg = reg
        self.solutions = None

        # HANDLE INITIAL SOLUTIONS
        inject_solutions = None
        if x0 is None:
            x0 = np.zeros(self.num_params)

        elif isinstance(x0, np.ndarray):
            x0 = np.atleast_2d(x0)
            inject_solutions = x0
            x0 = np.mean(x0, axis=0)

        # INITIALIZE
        import cma
        self.cma = cma.CMAEvolutionStrategy(x0, self.sigma_init, inopts)

        if inject_solutions is not None:
            if len(inject_solutions) == self.popsize:
                self.flush(inject_solutions)
            else:
                self.inject(inject_solutions)  # INJECT POTENTIALLY PROVIDED SOLUTIONS

    def inject(self, solutions=None):
        if solutions is not None:
            self.cma.inject(solutions, force=True)

    def flush(self, solutions):
        self.cma.ary = solutions
        self.solutions = solutions

    def rms_stdev(self):
        sigma = self.cma.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.cma.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = np.array(reward_table_result)
        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg
        self.cma.tell(self.solutions, (-reward_table).tolist())  # convert minimizer to maximizer.

    def current_param(self):
        return self.cma.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.cma.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.cma.result
        return r[0], -r[1], -r[1], r[6]


class PruneCMAES(CMAES):
    def __init__(self,
                 num_params,
                 sigma_init=1.0,
                 popsize=255,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 inopts=None,
                 sparsity=0.8,
                 prune_interval=2,
                 prune_ratio=0.1,
                 ):
        """Constructs a CMA-ES solver, based on Hansen's `cma` module which successively prunes parts of the
           parameters (to values of zero) that have the lowest absolute value.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param popsize: population size.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param inopts: dict-like CMAOptions, forwarded to cma.CMAEvolutionStrategy constructor).
        :param x0: (Optional) either (i) a single or (ii) several initial guesses for a good solution,
                   defaults to None (initialize via `np.zeros(num_parameters)`).
                   In case (i), the population is seeded with x0.
                   In case (ii), the population is seeded with mean(x0, axis=0) and x0 is subsequently injected.
        :param sparsity
        :param prune_interval:
        :param prune_ratio:
        """
        # pruning variables
        self.sparsity = sparsity
        self.prune_interval = prune_interval
        self.prune_ratio = prune_ratio

        CMAES.__init__(self, num_params=num_params, sigma_init=sigma_init, popsize=popsize,
                       weight_decay=weight_decay, reg=reg, x0=x0, inopts=inopts)

        # helpers
        self.step = 0
        self.check_step = 0
        self.current_reward_table = None

    @property
    def sparsity_level(self):
        return self.sparsity * min([1., self.prune_ratio * (self.step // self.prune_interval)])

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.cma.ask())
        self.solutions = self.prune()
        return self.solutions

    @property
    def step_pruning(self):
        max_pruning = min([self.num_params, np.rint(self.sparsity * self.num_params)])
        step_pruning = min([int(np.rint(self.prune_ratio * self.num_params) * (self.step // self.prune_interval)),
                            int(max_pruning)])
        return step_pruning

    def prune(self, threshold=None):
        step_pruning = self.step_pruning
        self.step += 1

        if step_pruning:
            # find a total number of `step_pruning` parameters with the largest variance amongst all solutions
            param_var = np.var(self.solutions, axis=0)
            threshold_var = np.sort(param_var)[::-1][step_pruning]
            prune_var = np.where(param_var >= threshold_var)[0]

            # find a total number of `step_pruning` parameters with the smallest mean (least bias) amongst all solutions
            param_abs_mean = np.abs(self.solutions.mean(axis=0))
            threshold_mean = sorted(param_abs_mean)[step_pruning]
            prune_mean = np.where(param_abs_mean <= threshold_mean)[0]

            # chose the intersection of large variance low mean to prune (set exactly to 0)
            prune_population = np.intersect1d(prune_var, prune_mean)
            self.solutions[:, prune_population] = 0

            # prune individual solutions whose parameters are among the number of `step_pruning` smallest
            if len(prune_population) < step_pruning:
                for i in range(self.popsize):
                    prune_individual = np.argsort(np.abs(self.solutions[i]))[:step_pruning]
                    self.solutions[i, prune_individual] = 0.

            self.flush(self.solutions)

        # Return the pruned parameters
        return self.solutions

    def tell(self, reward_table_result):
        self.check_sparsity_level(reward_table_result)
        return CMAES.tell(self, reward_table_result)

    def check_sparsity_level(self, reward_table_result):
        self.check_step += 1  # needs to be called before prune
        if not self.check_step % self.prune_interval:
            self.check_step = 0
            if self.current_reward_table is None or \
                    np.mean(reward_table_result) > np.mean(self.current_reward_table) or \
                    np.max(reward_table_result) > np.max(self.current_reward_table):
                self.current_reward_table = reward_table_result

            elif self.step >= 2*self.prune_interval:  # pruned solution for steps > 2 * pruning_interval
                # neither mean nor max did improve, reduce set sparsity level
                self.step -= 2*self.prune_interval


class SimpleGA:
    '''Simple Genetic Algorithm.'''

    def __init__(self, num_params,
                 sigma_init=1.0,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 popsize=256,
                 elite_ratio=0.1,
                 forget_best=False,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 ):
        """ Constructs a simple genetic algorithm instance.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param popsize: population size.
        :param elite_ratio: percentage of the elites.
        :param forget_best: forget the historical best elites.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(np.ceil(self.popsize * self.elite_ratio))

        # ADDING option to start from prior solution
        x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0)
        self.elite_params = np.stack([x0] * self.elite_popsize)
        # self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.best_param = np.copy(self.elite_params[0])  # np.zeros(self.num_params)

        self.sigma = self.sigma_init
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.reg = reg

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def flush(self, solutions):
        self.elite_params = solutions

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_param, self.best_reward, self.curr_best_reward, self.sigma


class OpenES:
    ''' Basic Version of OpenAI Evolution Strategies.'''

    def __init__(self, num_params,
                 sigma_init=1.0,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 learning_rate=0.01,
                 learning_rate_decay=0.9999,
                 learning_rate_limit=0.001,
                 popsize=256,
                 antithetic=False,
                 weight_decay=0.01,
                 reg='l2',
                 rank_fitness=True,
                 forget_best=True,
                 x0=None,
                 ):
        """ Constructs an evolutionary solver instance following the OpenAI algorithm.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param learning_rate: learning rate for standard deviation.
        :param learning_rate_decay: annealing the learning rate.
        :param learning_rate_limit: stop annealing learning rate.
        :param popsize: population size.
        :param antithetic: whether to use antithetic sampling.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param rank_fitness: use rank rather than fitness numbers.
        :param forget_best: forget historical best
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)

        # BH: ADDING option to start from prior solution
        self.mu = np.zeros(self.num_params) if x0 is None else np.asarray(x0)  # np.zeros(self.num_params)
        self.best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)

        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.reg = reg
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward += reg

        idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1. / (self.popsize * self.sigma) * np.dot(self.epsilon.T, normalized_reward)

        # self.mu += self.learning_rate * change_mu

        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def flush(self, solutions):
        self.solutions = solutions

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


class PEPG:
    '''Extension of PEPG with bells and whistles.'''

    def __init__(self, num_params,
                 sigma_init=1.0,
                 sigma_alpha=0.20,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 sigma_max_change=0.2,
                 learning_rate=0.01,
                 learning_rate_decay=0.9999,
                 learning_rate_limit=0.01,
                 elite_ratio=0,
                 popsize=256,
                 average_baseline=True,
                 weight_decay=0.01,
                 reg='l2',
                 rank_fitness=True,
                 forget_best=True,
                 x0=None,
                 ):  #
        """ Constructs a `PEPG` solver instance.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_alpha: learning rate for standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param sigma_max_change: clips adaptive sigma to 20%.
        :param learning_rate: learning rate for standard deviation.
        :param learning_rate_decay: annealing the learning rate.
        :param learning_rate_limit: stop annealing learning rate.
        :param elite_ratio: if > 0, then ignore learning_rate.
        :param popsize: population size.
        :param average_baseline: set baseline to average of batch.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param rank_fitness: use rank rather than fitness numbers.
        :param forget_best: don't keep the historical best solution.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu, rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)

        # BH: ADDING option to start from prior solution
        self.mu = np.zeros(self.num_params) if x0 is None else np.asarray(x0)  # np.zeros(self.num_params)
        self.best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)
        self.curr_best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)

        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.reg = reg
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon, then negative epsilon
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if (self.sigma_alpha > 0):
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if (self.sigma_decay < 1):
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def flush(self, solutions):
        self.solutions = solutions

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


def get_es_solver(name: str, return_signature: bool = False):
    if isinstance(name, str):
        if name not in globals().keys():
            raise NameError(f"Can't relate `{name}` with evolutionary solver.")

        cls = globals()[name]
    else:
        assert isinstance(name, type)
        cls = name

    if return_signature:
        import inspect
        signature = inspect.signature(cls)
        return cls, signature

    return cls


class BONES:
    ''' BottleNeck Evolutionary Strategy '''

    def __init__(self,
                 num_params,
                 bottleneck,
                 sigma_init=1.0,
                 popsize=256,
                 elitism=False,
                 elite_ratio=0.1,
                 mutable_genome=True,
                 mutation_ratio=0.1,
                 mutation_rule="standard_cauchy",
                 mutation_params=1,
                 roulette=0.,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 mating=False,
                 fitness=None,
                 mate_trials=1,
                 ):
        """ Constructs a BottleNeck Evolutionary Strategy instance.

        :param num_params: number of model parameters.
        :param bottleneck: A variational auto encoder type of bottleneck module which is used to transform phenotypes
                           to (offspring) genotypes.
        :param sigma_init: initial standard deviation.
        :param popsize: population size.
        :param elitism: Boolean flag to turn on/off elitism, i.e., protecting the elite parameters from extinction.
        :param elite_ratio: percentage of the elites which will produce offspring in the next generation.
        :param roulette: Optional numerical value which can be used - if different from 0 - to employ roulette-wheel
                         selection amongst the elites. Otherwise, randomly selected elites will reproduce.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        :param mating: Boolean flag to turn on/off mating procedure in genome space, defaults to False.
        :param fitness: Optional fitness module which can be used to find promising crossover points during mating.
        :param mate_trials: Optional positive total number of trials to find best performing crossover matings (i.e.,
                            loci to merge parent genomes) based on fitness prediction by `fitness` module.
        :param mutable_genome: Flag to enable/disable mutations in the genome after reproduction, defaults to True.
        :param mutation_ratio: Float specifying the fraction of the population undergoing mutations.
        :param mutation_rule: A string identifier for the numpy random process used to sample the noise,
                              e.g., "randn", "standard_cauchy", defaults to "standard_cauchy".
        :param mutation_params: Number of parameters that are mutated (at maximum) in the entire parameter sequence
                                of a genome undergoing mutation. If a negative value is provided, all paramters are
                                mutated, if 0 is provided, none is. Defaults to -1 (all are mutated).
        """

        self.num_params = num_params
        self.bottleneck = bottleneck

        self.sigma_init = sigma_init
        self.popsize = popsize

        self.sigma = self.sigma_init
        self.weight_decay = weight_decay
        self.reg = reg

        # population
        self.solutions = None

        # fertile population
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(np.ceil(self.popsize * self.elite_ratio))
        self.roulette = np.abs(roulette)
        self.elitism = elitism

        # crossover variables
        self.mating = mating
        self.fitness = fitness
        self.mate_trials = mate_trials

        # init fertile population
        if x0 is None:
            self.elite_params = np.random.randn(self.elite_popsize, self.num_params) * self.sigma_init
        else:
            x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0)
            if x0.ndim == 1:
                self.elite_params = np.stack([x0] * self.elite_popsize)
                self.elite_params[1:] += np.random.randn(*self.elite_params[1:].shape) * self.sigma_init
            elif len(x0) < self.elite_popsize:
                self.elite_params = np.stack([x0] * self.elite_popsize)[:self.elite_popsize]
                self.elite_params[len(x0):] += np.random.randn(*self.elite_params[len(x0):].shape) * self.sigma_init
            else:
                self.elite_params = np.asarray(x0)
            self.elite_params = self.elite_params[:self.elite_popsize]

        # helpers
        self.best_param = None
        self.best_reward = -np.inf
        self.curr_best_reward = None
        self.elite_rewards = None

        self.mutable_genome = mutable_genome
        self.mutation_ratio = mutation_ratio
        if isinstance(mutation_rule, str):
            self.mutation_rule = getattr(np.random, mutation_rule)

        if mutation_params < 0:
            mutation_params = num_params
        self.mutation_params = min([mutation_params, num_params])

    @property
    def step_size(self):
        return self.sigma_init

    def copy_genome(self, x):
        return x.copy() if hasattr(x, 'copy') else x.clone()

    def ask(self):
        """ returns a list of parameter. """
        elite_range = range(self.elite_popsize)
        p = None
        if self.roulette and np.any(self.elite_rewards):
            r_min, r_max = np.min(self.elite_rewards), np.max(self.elite_rewards)
            p = np.exp(-self.roulette*(self.elite_rewards - r_max) / (r_max - r_min))
            i = p.argsort()
            p[i] /= np.cumsum(p[i])
            p /= np.sum(p)

        selection = np.random.choice(elite_range, self.popsize * (1 + bool(self.mating)), p=p)
        genomes = self.elite_params[selection]

        # if self.mutable_genome:
        parents = self.bottleneck.decode(genomes)  # could be done via parameters arg in tell
        offspring_genomes = self.bottleneck.encode(parents)
        # else:
        #     offspring_genomes = self.copy_genome(genomes)
        #     if self.fitness is not None:
        #         from torch import tensor, double
        #         offspring_genomes = tensor(offspring_genomes, device=self.fitness.device, dtype=self.fitness.dtype)

        if self.mating:
            num_offspring = self.popsize
            parents = []
            if isinstance(self.mating, float):
                num_offspring = min([int(np.rint(self.popsize * self.mating)), self.popsize])
                if num_offspring < self.popsize:
                    parents = offspring_genomes[2*num_offspring:num_offspring + self.popsize]
            a, b = offspring_genomes[:num_offspring], offspring_genomes[num_offspring:2*num_offspring]
            offspring_genomes = self.mate(a, b)
            if self.fitness is not None and self.mate_trials > 1:
                f = self.fitness(offspring_genomes).flatten()
                for _ in range(self.mate_trials - 1):
                    c = self.mate(a, b)
                    f_c = self.fitness(c).flatten()  # (pop_size, 1) -> (pop_size)
                    fitter = f_c > f
                    offspring_genomes[fitter] = c[fitter]
                    f[fitter] = f_c[fitter]

            if len(parents):
                from torch import concatenate
                offspring_genomes = concatenate([offspring_genomes, parents], dim=0)

        if self.mutable_genome:
            offspring_genomes = self.mutate(offspring_genomes)

        self.solutions = offspring_genomes
        return offspring_genomes

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
        reward_table = np.array(reward_table_result)

        solution = self.solutions
        reward = reward_table
        if not isinstance(solution, np.ndarray):
            from mindcraft.torch.util import tensor_to_numpy
            solution = tensor_to_numpy(solution)

        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, solution, reg=self.reg)
            reward += reg

        if self.elitism and self.elite_rewards is not None:
            reward = np.concatenate([reward, self.elite_rewards])
        if self.elitism and self.elite_params is not None:
            solution = np.concatenate([solution, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]
        if self.curr_best_reward > self.best_reward:
            self.best_reward = self.curr_best_reward
            self.best_param = np.copy(self.elite_params[0])

    def current_param(self):
        return self.elite_params[0]

    def best_param(self):
        return self.best_param

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_param, self.best_reward, self.curr_best_reward, self.sigma

    def inject(self, solutions=None):
        raise NotImplementedError("inject")

    def flush(self, solutions):
        self.elite_params = solutions
        self.elite_rewards = None

    def mutate(self, o):
        """ mutate parts of the entire population with specified mutation ratio """
        # pick fraction of offspring for mutation
        mutable = np.random.choice(np.arange(len(o)),  # indices of offsprings
                                   int(np.rint(len(o) * self.mutation_ratio)),  # number of indiv. to mutate
                                   replace=False)  # no double-draw

        for i in mutable:  # mutate selected genomes
            # p = np.arange(self.num_params)
            p = np.arange(o.shape[1])
            if self.mutation_params != len(p):
                # pick only fraction of params to mutate, if `mutation_params` is defined
                p = np.unique(np.random.choice(p, self.mutation_params))

            mutations = self.mutation_rule(len(p)) * self.step_size
            try:
                o[i, p] += mutations
            except (TypeError, RuntimeError):
                o[i, p] += type(o)(mutations).to(o.device)  # PyTorch

        return o

    def mate(self, p_1, p_2):
        """ mate entire population at once (column wise crossover of p_1 and p_2) """
        child_params = self.copy_genome(p_2)
        p1_idx = np.random.rand(*tuple(p_1.shape)) > 0.5
        p1_idx = np.where(p1_idx)
        child_params[p1_idx] = p_1[p1_idx]
        return child_params


class MCMC:
    """ A Markov-Chain MontaCarlo solver """
    def __init__(self,
                 num_params,
                 sigma_init=1.0,
                 popsize=1,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 num_sweeps=None,
                 block_size=-1,
                 beta=np.inf,
                 beta_decay=1.01,
                 num_equilibrate=10,
                 minimize=False,
                 ):
        """ Constructs a `MCMC` instance

        :param num_params: number of parameters to be optimized.
        :param sigma_init: initial standard deviation.
        :param popsize: The number of statistically independent samples that are optimized in parallel.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param x0: An initial value for the optimization, must be either a 1d or 2d array.
                   If a 1d array is provided, one element of a number of `popsize` statistically independent samples
                   will be set to the initial value, and the rest will be subjected to additional Gaussian noise
                   scaled by sigma init.
                   If a 2d array of N parameter sets is provided, the first N elements of the number of `popsize`
                   statistically independent samples are set by `x0`, the rest are Gaussian noise samples scaled by
                   sigma init.
        :param num_sweeps: Number of sweeps before a sample is considered statistically independent, defaults
                           to `None` (which translates to `num_params / block_size`).
        :param block_size: The number of randomly chosen parameters that are updated at a single sweep,
                           defaults to `-1` which updates all parameter by adding Gaussian noise scaled by `sigma_init`.
        :param beta: Inverse temperature `beta = 1/T` used for metropolis sampling, defaults to `inf`,
                     i.e., zero temperature and thus greedy search.
        :param beta_decay: Decay rate for `beta *= beda_decay`, defaults to `1.01`.
        :param num_equilibrate: Number of steps to equilibrate the samples at a given `beta`,
        :param minimize: Boolean flag to enable minimzation optimization, defaults to `false`.
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.reg = reg
        self.x0 = x0
        self.block_size = block_size if block_size > 0 else num_params
        self.num_sweeps = num_sweeps or int(num_params // self.block_size)
        self.beta = beta
        self.beta_decay = beta_decay
        self.num_equilibrate = num_equilibrate
        self.minimize = minimize

        self.solutions = np.random.randn(popsize, num_params) * sigma_init
        if x0 is not None:
            if np.ndim(x0) == 1:
                self.solutions[0, :] = x0
                self.solutions[1:, :] += x0

            elif np.ndim(x0) == 2:
                self.solutions[:min((len(x0), popsize)), :] = x0[:min((len(x0), popsize))]

            else:
                raise ValueError(f"Only 1d or 2d shapes allowed for `x0`, got {np.shape(x0)}d.")

        self._sweeps, self._sweep_ids, self._eval = None, None, None
        self.info = {'sweep': [], 'eval': [], 'beta': []}
        self._eval = np.full(popsize, fill_value=np.inf)
        self._step = 0

        self.best_solution = None
        self.best_eval = np.inf

    def ask(self):
        '''returns a list of parameters'''
        self._sweeps, self._sweep_ids = self.sweep()
        return self._sweeps

    def sweep(self):
        if self.block_size < self.num_params:
            sweep_ids = np.stack([np.random.choice(self.num_params, size=self.block_size, replace=False)
                                  for _ in range(self.popsize)])
        else:
            sweep_ids = np.repeat(np.arange(0, self.num_params)[None, :], self.popsize, axis=0)

        sweeps = self.solutions.copy()
        rands = np.random.randn(*sweep_ids.shape) * self.sigma
        for i, (s, r) in enumerate(zip(sweep_ids, rands)):
            sweeps[i, s] += r

        return sweeps, sweep_ids

    @property
    def sigma(self):
        return self.sigma_init

    def tell(self, reward_table_result):
        max2min = -1.**(not self.minimize)
        reward_table = np.array(reward_table_result) * max2min

        if self.weight_decay > 0:
            reg = compute_weight_decay(self.weight_decay, self._sweeps, reg=self.reg)
            reward_table += reg * max2min

        accept = self.detailed_balance(old_value=self._eval, new_value=reward_table, beta=self.beta)
        self.solutions[accept] = self._sweeps[accept]
        self._eval[accept] = reward_table[accept]

        current_best_eval = np.min(self._eval)
        if self.best_eval > current_best_eval:
            self.best_eval = current_best_eval
            self.best_solution = self.solutions[np.argmin(self._eval)]

        self.info['eval'].append(self._eval.tolist())
        self.info['sweep'].append(accept.tolist())
        self.info['beta'].append(self.beta)

        self._step += 1
        if not self._step % (self.num_sweeps * self.num_equilibrate):
            self.beta *= self.beta_decay
            # print(f"### beta: {self.beta}, accepted: {np.mean(self.info['sweep'][-self.num_sweeps * self.num_equilibrate:])}")
            # print("############################")

    @staticmethod
    def detailed_balance(old_value: np.ndarray, new_value: np.ndarray, beta: np.ndarray):
        if np.isinf(beta):  # zero_temp
            accept = (old_value > new_value).flatten()

        elif beta == 0:  # inf temp.
            accept = np.ones(len(old_value), dtype=bool)

        else:
            # return numpy.random.rand() > min(1., numpy.exp((new_evaluation-old_evaluation)*beta))
            rands = np.random.rand(len(old_value))
            crit = (rands < np.minimum(1., np.exp(-(new_value - old_value) * beta)))
            accept = (crit == 1)

        return accept

    @property
    def curr_best_solution(self):
        best_idx = np.argmin(self._eval)
        return self.solutions[best_idx]

    @property
    def curr_best_reward(self):
        return np.argmin(self._eval) * (-1 ** (not self.minimize))

    @property
    def best_reward(self):
        return self.best_eval * (-1 ** (not self.minimize))

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_solution, self.best_reward, self.curr_best_reward, self.sigma)
