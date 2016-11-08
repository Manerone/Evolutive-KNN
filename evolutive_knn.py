from sklearn.neighbors import KNeighborsClassifier
from individual import Individual
import random
# from numba import jit

class EvolutiveKNN:
    """Implementation of an evolutive version of KNN.
    This class finds the best K and the best weigths for a given training set

    Parameters:
        describe parameters here
    
    Usage:
        give usage examples
    """
    def __init__(self, population_size=100, mutation_rate=0.02, max_generations=50, max_accuracy=0.95, max_k=None, max_weight=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_accuracy = max_accuracy
        self.max_k = max_k
        self.max_weight = max_weight

    """This method is responsible for training the evolutive KNN based on the
    given training_examples and training_labels

    Parameters:
        training_examples: Array of features, each feature is an array of floats
            example: [[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]]
        training_labels: Array of integers that are the labels for each feature.
            example: [0, 1, 0]
        Observation: the first label is the class of the first feature, and so on.

    Usage:
        classifier = EvolutiveKNN()
        classifier.train([[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]], [0, 1, 0])
    """
    def train(self, training_examples, training_labels):
        population = self._start_population(len(training_labels))
        # calculo fitness de todos
        # enquanto nao tiver satisfeito condicao de parada
        #   realizo crossovers
        #   realizo mutacoes
        #   realizo elitismo
        #   calculo fitness da nova populacao

    def _start_population(self, maximum_k):
        population = []
        population_size = self.population_size
        if self.max_k is None:
            self.max_k = maximum_k
        max_k = self.max_k
        max_weight = self.max_weight
        for _ in xrange(population_size):
            k = random.randint(1, max_k)
            weights = [random.choice(range(max_weight)) for _ in xrange(k)]
            population.append(
                Individual(k, weights)
            )
        return population