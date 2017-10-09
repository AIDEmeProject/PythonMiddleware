import numpy as np

from src.utils import check_points

def get_constrain(kind, vector=None, matrix=None):
    if vector is None:
        return EmptyConstrain()
    elif kind == 'lower':
        return LowerConstrain(vector)
    elif kind == 'upper':
        return UpperConstrain(vector)
    elif kind == 'equality':
        return EqualityConstrain(vector, matrix)
    elif kind == 'inequality':
        return InequalityConstrain(vector, matrix)
    else:
        raise ValueError("Unknown kind. Only 'equality', 'inequality', 'lower' and 'upper' supported.")


class Constrain:
    _kind = None

    def __init__(self, vector, matrix=None):
        self._vector = np.asarray(vector, dtype=np.float64).ravel()

        if self.kind in ['equality', 'inequality']:
            self._matrix = check_points(matrix)
            if len(self._vector) != len(self._matrix):
                raise ValueError("Incompatible sizes.")

    def __len__(self):
        return len(self._vector)

    def is_empty(self):
        return len(self._vector) == 0

    @property
    def matrix(self):
        if hasattr(self, '_matrix'):
            return self._matrix

    @property
    def vector(self):
        return self._vector

    @property
    def shape(self):
        if hasattr(self, '_matrix'):
            return self.matrix.shape

        return (len(self), )

    @property
    def kind(self):
        return self._kind

    def check(self, points):
        raise NotImplementedError


class EmptyConstrain(Constrain):
    _kind = 'empty'

    def __init__(self):
        pass

    def __len__(self):
        return 0

    def is_empty(self):
        return True

    def check(self, points):
        return True


class EqualityConstrain(Constrain):
    _kind = 'equality'

    def check(self, points):
        return np.allclose(np.dot(points, self._matrix.T), self._vector)

class InequalityConstrain(Constrain):
    _kind = 'inequality'

    def check(self, points):
        return np.all(np.dot(points, self.matrix.T) <= self.vector, axis=-1)

class LowerConstrain(Constrain):
    _kind = 'lower'

    def check(self, points):
        return np.all(points >= self._vector, axis=-1)


class UpperConstrain(Constrain):
    _kind = 'upper'

    def check(self, points):
        return np.all(points <= self._vector, axis=-1)
