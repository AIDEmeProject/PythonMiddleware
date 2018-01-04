import numpy as np
from scipy.special import gamma

from src.main.utils import check_sizes
from .base import ConvexBody


def solve_second_degree_equation(a, b, c):
    """ Solves the equation a x^2 + 2bx + c = 0 """
    delta = b**2 - a*c
    if delta <= 0:
        raise ValueError("Equation has no real solutions")
    return (-b - np.sqrt(delta))/a, (-b + np.sqrt(delta))/a


class Ellipsoid(ConvexBody):
    """
    Elipsoid centered at C and with half-axis lengths (l_1, ..., l_n)
    """

    def __init__(self, center, half_axis_length):
        check_sizes(center, half_axis_length)

        if not all(map(lambda x: x > 0, half_axis_length)):
            raise ValueError("Negative axis length!")

        super().__init__(len(center))

        self._center = np.asarray(center, dtype=np.float64).ravel()
        self._half_axis_length = np.asarray(half_axis_length, dtype=np.float64).ravel()



    def __repr__(self):
        return "Center: {center}\nHalf-axis length: {length}".format(center=self.center, length=self.half_axis_length)

    @property
    def center(self):
        return self._center

    @property
    def half_axis_length(self):
        return self._half_axis_length

    @property
    def volume(self):
        factor = np.prod(self.half_axis_length)
        return factor * pow(np.pi, self._dim / 2.0)/ gamma(1 + self._dim / 2.0)

    def sample(self, n_samples):
        samples = np.random.normal(size=(n_samples, self._dim))
        samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
        samples *= np.power(np.random.uniform(low=0, high=1, size=(n_samples, 1)), 1.0 / self.dim)
        return self.center + self.half_axis_length * samples

    def is_inside(self, points):
        normalized_points = (points - self.center) / self.half_axis_length
        return np.sum(normalized_points ** 2, axis=-1) < 1

    def intersection(self, line):
        normalized_point = (line.center - self.center)/self.half_axis_length
        normalized_direction = line.direction/self.half_axis_length
        
        a, b, c = (
            np.sum(normalized_direction ** 2), 
            normalized_point.dot(normalized_direction), 
            np.sum(normalized_point ** 2) - 1
        )
        intersection1, intersection2 = solve_second_degree_equation(a, b, c)

        return line.get_segment(intersection1, intersection2)



class Ball(Ellipsoid):
    """ Ball centered at C and of radius R """

    def __init__(self, center, radius):
        ConvexBody.__init__(self)

        self._center = np.asarray(center, dtype=np.float64).ravel()
        self._dim = len(self._center)
        
        self._radius = float(radius)
        if self._radius <= 0:
            raise ValueError("Negative radius found.")

    def __repr__(self):
        return "Center: {center}\nRadius: {radius}".format(center=self.center, radius=self.radius)

    @property
    def radius(self):
        return self._radius

    @property
    def half_axis_length(self):
        return np.ones(self._dim) * self._radius


class UnitBall(Ball):
    def __init__(self, dim):
        ConvexBody.__init__(self)

        self._dim = int(dim)
        if self._dim <= 0:
            raise ValueError("Received non-positive dimensions.")

        self._center = np.zeros(self._dim)
        self._radius = 1.0
