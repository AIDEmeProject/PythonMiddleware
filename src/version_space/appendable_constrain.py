import numpy as np
from ..convexbody.objects.constrain import InequalityConstrain


class AppendableInequalityConstrain(InequalityConstrain):
    def __init__(self):
        self._vector = []
        self._matrix = []

    def __len__(self):
        return len(self._vector)

    @property
    def vector(self):
        return np.array(self._vector)

    @property
    def matrix(self):
        return np.array(self._matrix)

    @property
    def shape(self):
        if self.is_empty():
            return (0,)
        else:
            return (len(self._matrix), len(self._matrix[0]))

    def _check_sizes(self, point):
        if not self.is_empty() and point.shape != (self.shape[1], ):
            raise ValueError("Bad point dimension: obtained {0}, expected {1}".format(point.shape, self.shape))

    def clear(self):
        self._vector = []
        self._matrix = []

    def append(self, point, value):
        value = float(value)
        point = np.asarray(point).ravel()
        self._check_sizes(point)

        self._vector.append(value)
        self._matrix.append(point)

    def check(self, points):
        return True if self.is_empty() else super().check(points)