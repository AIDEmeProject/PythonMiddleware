import numpy as np

from .base import ConvexBody


class Box(ConvexBody):
    """ 
    Box [a_1, b_1] x ... x [a_n, b_n]
    """
    
    def __init__(self, low, high):
        """
        :param low: list [a_1, ..., a_n] 
        :param high: list [b_1, ..., b_n] 
        """
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        assert np.all(self.low < self.high), "Negative side length!"
        
        self.n = self.low.shape[0]
        self.L = self.high - self.low
        
    def is_inside(self, x):
        return np.all(np.logical_and(x <= self.high, x >= self.low), axis=-1)
            
    def volume(self):
        return np.prod(self.L)

    def sample(self, n_samples):
        return self.low + self.L * np.random.uniform(size=(n_samples, self.n))


class Cube(Box):
    """ 
    Cube centered at C and with side length L.
    """
    
    def __init__(self, center, length):
        assert length > 0, "Received non-positive length!"
        self.C = np.asarray(center, dtype=np.float32)
        super().__init__(low=self.C - 0.5*length, high=self.C + 0.5*length)
        self.L = length

    def is_inside(self, x):
        return np.all(np.abs(x-self.C) <= 0.5*self.L, axis=-1)

    def volume(self):
        return pow(self.L, self.n)


class StandardCube(Cube):
    """
    Cube centered at the origin and of side 2
    """

    def __init__(self, n):
        assert isinstance(n, int) and n > 0, "Dimension n must be a positive integer."

        super().__init__(center=np.zeros(n), length=2.0)

    def sample(self, n_samples):
        return np.random.uniform(low=-1, high=1, size=(n_samples, self.n))
