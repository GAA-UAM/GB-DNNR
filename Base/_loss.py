""" Loss functions for Gradient Boosted - Deep Neural Network - Multi Output """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from abc import abstractmethod



class loss:
    """ Template class for the loss function """

    @abstractmethod
    def model0(self, y):
        """abstract method for initialization of approximation"""

    @abstractmethod
    def derive(self, y, prev):
        """abstract method for derive"""

    @abstractmethod
    def __call__(self, y, pred):
        """call"""


class squared_loss(loss):
    """ Squared loss for regression problems """

    def model0(self, y):
        return np.ones(1)*np.mean(y)

    def derive(self, y, prev):
        return y-prev

    def __call__(self, y, pred):
        return (y-pred)**2
