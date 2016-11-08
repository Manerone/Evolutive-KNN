class Individual:
    def __init__(self, k, weights, fitness=0.0):
        if k != len(weights):
            raise Exception('K and weights size are diferent')
        self.k = k
        self.weights = weights
        self.fitness = fitness
    
    def set_fitness(self, fitness):
        self.fitness = fitness