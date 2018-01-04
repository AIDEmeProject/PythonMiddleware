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
        super().__init__(len(low))
        
        self._low = np.array(low, dtype=np.float64)
        self._high = np.array(high, dtype=np.float64)

        if np.any(self._low >= self._high):
            raise ValueError("Non-positive side length!")

        self._center = 0.5 * (self._low + self._high)
        self._side_length = self._high - self._low

    def __repr__(self):
        return "Low: {low}\nHigh: {high}\nCenter: {center}\nSide_length: {length}".format(
            low=self.low,
            high=self.high,
            center=self.center,
            length=self.side_length
        )

    @property
    def low(self):
        return self._low
    
    @property
    def high(self):
        return self._high
    
    @property
    def center(self):
        return self._center
    
    @property
    def side_length(self):
        return self._side_length

    def is_inside(self, points):
        centered_points = np.abs(points - self.center)
        return np.all(centered_points < 0.5 * self.side_length, axis=-1)

    @property
    def volume(self):
        return np.prod(self.side_length)

    def sample(self, n_samples):
        samples_over_standard_cube = np.random.uniform(low=-1, high=1, size=(n_samples, self.dim))
        return self.center + 0.5 * self.side_length * samples_over_standard_cube

    def intersection(self, line):
        left, right = (self.low - line.center) / line.direction, (self.high - line.center) / line.direction

        lower_bounds = np.hstack([left[line.direction > 0], right[line.direction < 0]])
        upper_bounds = np.hstack([left[line.direction < 0], right[line.direction > 0]])

        if len(lower_bounds) == 0 or len(upper_bounds) == 0:
            raise RuntimeError("Line does not intersect box.")

        t1, t2 = np.max(lower_bounds), np.min(upper_bounds)

        if t1 >= t2:
            raise RuntimeError("Line does not intersect box.")

        return line.get_segment(t1, t2)


class Cube(Box):
    """
    Cube centered at "center" and with side equal to "length".
    """

    def __init__(self, center, length):
        ConvexBody.__init__(self)

        self._center = np.asarray(center, dtype=np.float64)

        self._side_length = float(length)
        if self._side_length <= 0:
            raise ValueError("Received non-positive length!")

        self._dim = len(self._center)

    @property
    def low(self):
        return self.center - 0.5 * self.side_length

    @property
    def high(self):
        return self.center + 0.5 * self.side_length

    @property
    def volume(self):
        return self.side_length ** self.dim


class UnitCube(Cube):
    def __init__(self, dim):
        ConvexBody.__init__(self)

        self._dim = int(dim)
        if self._dim <= 0:
            raise ValueError("Received non-positive dimension.")

    @property
    def center(self):
        return np.zeros(self.dim)

    @property
    def side_length(self):
        return 2.0
