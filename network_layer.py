import numpy as np
import random

class neural_layer():
    def __init__():
        pass

class neural_weight(object):
    def __init__(self, rows, columns, initializer = 'uniform'):
        if initializer == 'uniform':
            self.weight = np.random.uniform(0, 1.0, (rows, columns))
        elif initializer == 'normal':
            self.weight = np.random.randn(rows, columns)
        else:
            self.weight = np.zeros((rows, columns))

        self.delta = np.zeros((rows, columns))
        self.velocity = np.zeros((rows, columns))

