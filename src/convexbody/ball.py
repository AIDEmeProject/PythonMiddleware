import numpy as np
from scipy.special import gamma

from .base import ConvexBody


class Ellipsoid(ConvexBody):
    """
    Elipsoid centered at C and with half-axis lengths (l_1, ..., l_n)
    """
    
    def __init__(self, center, lengths):
        self.C = np.asarray(center, dtype=np.float64)
        self.R = np.asarray(lengths, dtype=np.float64)
        self.n = len(self.C)
        
        if not np.all(self.R > 0):
            raise ValueError("Negative axis length!")

        if not len(self.C) == len(self.R):
            raise ValueError("Wrong dimensions!")

    def is_inside(self, x):
        return np.sum(np.square((x - self.C)/self.R), axis=-1) <= 1

    def volume(self):
        return pow(np.pi, self.n / 2.0) * np.prod(self.R) / gamma(1 + self.n / 2.0)

    def sample(self, n_samples):
        samples = np.random.normal(size=(n_samples, self.n))
        samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
        samples *= np.power(np.random.uniform(low=0, high=1, size=(n_samples, 1)), 1.0 / self.n)
        return self.C + self.R * samples


class Ball(Ellipsoid):
    """ Ball centered at C and of radius R """
    
    def __init__(self, center, radius):  
        assert radius > 0, "Received non-positive radius!"
        
        self.C = np.asarray(center, dtype=np.float64)
        self.n = len(self.C)
        self.R = float(radius)
        
    def volume(self):
        return pow(np.pi, self.n/2.0)*pow(self.R, self.n)/gamma(1 + self.n/2.0)


class UnitBall(Ball):
    """
    Ball centered at the origin and of radius 1
    """

    def __init__(self, n):
        self.n = n

        super().__init__(center=np.zeros(self.n), radius=1.0)
