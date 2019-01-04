import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class DiagonalGaussianKernel:
    def __init__(self, D):
        self.D = np.array(D).reshape(1, -1)
        assert np.all(self.D > 0)
        self.D = np.sqrt(self.D)

    def __call__(self, X, Y=None):
        X = X * self.D

        if Y is None:
            Y = X
        else:
            Y = Y * self.D

        return rbf_kernel(X, Y, gamma=1.0)
