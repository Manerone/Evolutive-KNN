class Individual:
    def __init__(self, k, weights):
        if k != len(weights):
            raise Exception('K and weights size are diferent')
        self.k = k
        self.weights = weights