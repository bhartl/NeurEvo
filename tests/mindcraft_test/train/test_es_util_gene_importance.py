if __name__ == '__main__':
    import numpy as np
    import cma
    from sklearn.metrics import normalized_mutual_info_score

    # Define the fitness function
    def fitness_function(x):
        x1 = np.sum(1. - np.abs(x[0:int(len(x)//4)] - 1.))
        x2 = np.sum((0.25 < x[-int(len(x)//2):-int(len(x)//4)-1]) * (x[-int(len(x)//2):-int(len(x)//4)-1] < 0.5))
        x3 = np.sum(1. - np.abs(x[-int(len(x)//4):] + 1.))
        return - (x1 + x2 + x3)


    # Set the CMA-ES parameters
    pop_size = 20
    num_genes = 10
    sigma = 0.1
    bounds = [-1, 1]

    # Initialize the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(np.zeros(num_genes), sigma, {'bounds': bounds, 'popsize': pop_size}, )

    # Run the CMA-ES optimization
    while not es.stop():
        solutions = es.ask()
        fitness = [fitness_function(x) for x in solutions]
        es.tell(solutions, fitness)

    # Get the best solution and covariance matrix
    best_solution = es.best.get()[0]
    print(best_solution, fitness_function(best_solution))
    print(np.shape(solutions))

    covariance_matrix = es.C
    cov = np.cov(solutions, rowvar=False)

    # Calculate the Mahalanobis distance for each gene
    mahalanobis_distances = np.diag(np.dot(np.dot((solutions - es.mean), np.linalg.inv(covariance_matrix)).T, (solutions - es.mean)))

    num_bins = 1000
    discretized_solutions = np.floor(
        (solutions - np.min(solutions)) / (np.max(solutions) - np.min(solutions)) * num_bins).astype(int)

    mutual_information = [normalized_mutual_info_score(discretized_solutions[:, i], np.asarray(fitness) * 100.) for i in range(num_genes)]

    # Print the Mahalanobis distance for each gene
    for i, (md, mi) in enumerate(zip(mahalanobis_distances, mutual_information)):
        print("Gene {}: Mahalanobis distance = {}, Mutual information = {}".format(i, md, mi))

    import matplotlib.pyplot as plt
    plt.plot(covariance_matrix.diagonal(), marker='o')

    ax2 = plt.gca().twinx()
    ax2.plot(cov.diagonal(), marker='x')
    plt.show()
