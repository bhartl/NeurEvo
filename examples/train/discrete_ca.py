import numpy as np
from mindcraft.train.es_util import DiscreteGA


def objective_function(params, num_letters=2):
    """
    Example objective function that optimizes for a an alternating pattern of letters, eg `[0, 1, 0, 1, ...]` for 2 letters.
    This should be replaced with the actual problem-specific function.
    """

    num_params = params.shape[1]
    letters = np.array([i % num_letters for i in range(num_params)])
    return 1. - np.sum(np.abs(params - letters[None, :]), axis=-1) / num_params

def main_discrete(num_params=10, num_letters=3, num_generations=10, population_size=100):
    ga = DiscreteGA(
        num_params=num_params,
        num_letters=num_letters,
        popsize=population_size,
        mutation_rate=0.25,
        mutation_decay=1.,
    )

    xt = []
    ft = []
    for generation in range(num_generations):
        # ask the GA for a batch of parameters
        x = ga.ask()
        f = objective_function(x, num_letters=num_letters)

        # Tell the GA about the fitness
        ga.tell(f)
        xt.append(ga.best_param)
        ft.append(ga.best_reward)

        # Print status
        print(f"Generation {generation + 1}/{num_generations}, Best fitness: {np.max(f)}, Best historic: {ga.best_reward}")
        print(f"Best parameters: {ga.best_param}")
        print(f"Target letters: {[i % num_letters for i in range(num_params)]}")

    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(ft, label='Best Fitness')
    ax1.set_ylabel('Best Fitness')
    ax1.legend()
    ax2.imshow(np.array(xt).T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Parameter Index')
    ax2.set_title('Parameter Evolution Over Generations')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argh
    argh.dispatch_command(main_discrete)

