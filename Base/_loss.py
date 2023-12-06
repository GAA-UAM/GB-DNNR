import numpy as np


class squared_loss:
    def model0(self, y):
        return np.ones(1) * np.mean(y)

    def derive(self, y, prev):
        return y - prev

    def __call__(self, y, pred):
        return (y - pred) ** 2
