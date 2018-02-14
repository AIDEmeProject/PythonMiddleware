import numpy as np
from scipy.optimize import linprog
from ..convexbody.objects.constrain import InequalityConstrain


class AppendableInequalityConstrain(InequalityConstrain):
    def __init__(self, dim):
        self._vector = []
        self._matrix = []
        self._dim = int(dim)

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
            return (0, self._dim)
        else:
            return (len(self._matrix), self._dim)

    def _check_sizes(self, point):
        if point.shape != (self._dim, ):
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

    def get_point(self):
        """
            Finds an interior point to the current search space through an optimization routine.
            :return: point inside search space
        """
        n_constrains, dim = self.shape

        if self.is_empty():
            return np.zeros(dim)

        res = linprog(
            c=np.array([1.0] + [0.0] * dim),
            A_ub=np.hstack([-np.ones((n_constrains, 1)), self.matrix]),
            b_ub=np.zeros(n_constrains),
            bounds=[(None, None)] + [(-1, 1)] * dim
        )

        point = res.x[1:].ravel()

        if not self.check(point):
            raise RuntimeError("Linear Program optimization failed: {0} does not satisfy constrains.".format(point))

        if np.allclose(point, 0):
            raise RuntimeError("Found zero vector. Check constrains for degeneracy of Version Space.")

        return 0.99 * point / np.linalg.norm(point)

    def intersection(self, line):
        den = np.dot(self.matrix, line.direction)
        r = (self.vector - np.dot(self.matrix, line.center)) / den
        r1 = r[den < 0]
        r2 = r[den > 0]

        t1 = -float('inf') if len(r1) == 0 else np.max(np.hstack(r1))
        t2 = float('inf') if len(r2) == 0 else np.min(np.hstack(r2))

        if t1 >= t2:
            raise RuntimeError("Line does not intersect polytope.")

        return line.get_segment(t1, t2)
