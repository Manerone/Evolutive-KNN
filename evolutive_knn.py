from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from individual import Individual
import random
# from numba import jit

class EvolutiveKNN:
    """Implementation of an evolutive version of KNN.
    This class finds the best K and the best weigths for a given training set
    """
    """
    EvolutiveKNN initializer

    Parameters:
        training_examples: Array of features, each feature is an array of floats
            example: [[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]]
        training_labels: Array of integers that are the labels for each feature.
            example: [0, 1, 0]
        Observation: the first label is the class of the first feature, and so on.
    
    Usage:
        classifier = EvolutiveKNN([[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]], [0, 1, 0])
    """
    def __init__(self, training_examples, training_labels, ts_size = 0.5):
        test_size = int(ts_size * len(training_labels))
        self._create_test(
            np.array(training_examples), np.array(training_labels), test_size
        )

        print test_size
        print self.training_examples
        print self.training_labels
        print self.test_examples
        print self.test_labels

    """This method is responsible for training the evolutive KNN based on the
    given parameters
    """
    def train(self, population_size=100, mutation_rate=0.02, max_generations=50, max_accuracy=0.95, max_k=None, max_weight=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_accuracy = max_accuracy
        self.max_k = max_k
        self.max_weight = max_weight
        self.global_best = Individual(1, [1])
        self._train()

    def _train(self):
        population = self._start_population()
        self._calculate_fitness_of_population(population)
        generations = 1
        # enquanto nao tiver satisfeito condicao de parada
        #   realizo crossovers
        #   realizo mutacoes
        #   realizo elitismo
        #   calculo fitness da nova populacao

    def _start_population(self):
        max_k = self.max_k
        if max_k is None: max_k = len(self.training_labels)
        population = []
        for _ in xrange(self.population_size):
            k = random.randint(1, max_k)
            weights = [
                random.choice(range(self.max_weight)) for _ in xrange(k)
            ]
            population.append(Individual(k, weights))
        return population

    def _calculate_fitness_of_population(self, population):
        for index, element in enumerate(population):
            print "element: ", index
            self._calculate_fitness_of_individual(element)
            if self.global_best.fitness < element.fitness:
                self.global_best = element

    def _calculate_fitness_of_individual(self, element):

        def _element_weights(distances):
            return element.weights

        kneigh = KNeighborsClassifier(n_neighbors=element.k, weights=_element_weights)
        kneigh.fit(self.training_examples, self.training_labels)
        element.fitness = kneigh.score(self.test_examples, self.test_labels)

    def _create_test(self, tr_examples, tr_labels, test_size):
        self.training_examples = []
        self.training_labels = [] 
        self.test_examples = []
        self.test_labels = []

        test_indexes = random.sample(xrange(len(tr_labels)), test_size)

        self.test_examples = tr_examples[test_indexes]
        self.test_labels = tr_labels[test_indexes]
        for index in xrange(len(tr_labels)):
            if index not in test_indexes:
                self.training_examples.append(tr_examples[index])
                self.training_labels.append(tr_labels[index])
        self.training_examples = np.array(self.training_examples)
        self.training_labels = np.array(self.training_labels)