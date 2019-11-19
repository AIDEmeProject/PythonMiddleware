import numpy as np

from .convex import ConvexHull, ConvexCone, ConvexError
from .one_dim_convex import OneDimensionalConvexHull, OneDimensionalConvexCone
from aideme.utils import assert_positive


class Polytope:
    def __init__(self, tol=1e-12):
        self.positive_region = PositiveRegion(tol)
        self.negative_region = []
        self.__is_valid = True

    @property
    def is_valid(self):
        return self.__is_valid

    def clear(self):
        self.positive_region.clear()
        self.negative_region = []
        self.__is_valid = True

    def predict(self, X):
        """
            Predicts the label of each data point. Returns 1 if point is in positive region, 0 if in negative region,
            and 0.5 otherwise

            :param X: data matrix to predict labels
        """
        X = np.atleast_2d(X)

        probas = np.full(len(X), fill_value=0.5)

        if self.__is_valid:
            if self.positive_region.is_built:
                probas[self.positive_region.is_inside(X)] = 1.0

            for cone in self.negative_region:
                if cone.is_built:
                    probas[cone.is_inside(X)] = 0.0

        return probas

    def update(self, X, y):
        """
        Increments the polytope model with new labeled data

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.__is_valid:
            raise RuntimeError("Cannot update invalid polytope.")

        X, y = np.atleast_2d(X), np.asarray(y)

        try:
            self.__update_positive(X[y == 1])
            self.__update_negative(X[y != 1])

        except ConvexError:
            self.__is_valid = False

        finally:
            return self.__is_valid

    def __update_positive(self, X):
        if len(X) > 0:
            self.positive_region.update(X)

            for nr in self.negative_region:
                nr.add_points_to_hull(X)

    def __update_negative(self, X):
        for x in X:
            cone = NegativeCone(x, self.positive_region._tol)
            cone.add_points_to_hull(self.positive_region.vertices)
            self.negative_region.append(cone)


class PositiveRegion:
    def __init__(self, tol=1e-12):
        assert_positive(tol, 'tol')

        self._hull = None
        self._cache = Cache()
        self._tol = tol

    @property
    def is_built(self):
        return self._hull is not None

    @property
    def vertices(self):
        return self._cache.data if self._hull is None else self._hull.vertices

    def clear(self):
        self._hull = None
        self._cache.clear()

    def is_inside(self, X):
        if not self.is_built:
            return np.full(len(X), False)

        return self._hull.is_inside(X)

    def update(self, pos_points):
        if self.is_built:
            self._hull.add_points(pos_points)
            return

        self._cache.update(pos_points)

        if self._cache.size > self._cache.dim:
            self._hull = self.__build_convex_hull()
            self._cache.clear()

    def __build_convex_hull(self):
        if self._cache.dim > 1:
            return ConvexHull(self._cache.data, self._tol)

        return OneDimensionalConvexHull(self._cache.data)


class NegativeCone:
    def __init__(self, vertex, tol=1e-12):
        assert_positive(tol, 'tol')

        self._vertex = np.asarray(vertex).reshape(-1)
        self._cache = Cache()
        self._cone = None
        self._tol = tol

    @property
    def is_built(self):
        return self._cone is not None

    def clear(self):
        self._cone = None
        self._cache.clear()

    def is_inside(self, X):
        if not self.is_built:
            return np.full(len(X), False)

        return self._cone.is_inside(X)

    def add_points_to_hull(self, points):
        points = np.atleast_2d(points)

        if points.shape[1] != len(self._vertex):
            raise ValueError("Bad input dimension: expected {0}, got {1}".format(len(self._vertex), points.shape[1]))

        if self._cone is not None:
            self._cone.add_points_to_hull(points)
            return

        self._cache.update(points)

        if self._cache.size >= self._cache.dim:
            self._cone = self.__build_convex_cone()
            self._cache.clear()

    def __build_convex_cone(self):
        if len(self._vertex) > 1:
            return ConvexCone(self._cache.data, self._vertex, tol=self._tol)
        else:
            return OneDimensionalConvexCone(self._cache.data, self._vertex)


class Cache:
    def __init__(self):
        self.data = None

    @property
    def size(self):
        return 0 if self.data is None else self.data.shape[0]

    @property
    def dim(self):
        return -1 if self.data is None else self.data.shape[1]

    def clear(self):
        self.data = None

    def update(self, X):
        X = np.atleast_2d(X)

        if self.data is None:
            self.data = X.copy()
        else:
            self.data = np.vstack([self.data, X])
