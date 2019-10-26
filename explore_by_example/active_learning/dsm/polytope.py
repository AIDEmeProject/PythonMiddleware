import numpy as np

from .convex import ConvexHull, ConvexCone
from .one_dim_convex import OneDimensionalConvexHull, OneDimensionalConvexCone
from explore_by_example.utils import assert_positive


class PolytopeModel:
    def __init__(self, tol=1e-12):
        self.positive_region = PositiveRegion(tol)
        self.negative_region = NegativeRegion(self.positive_region, tol)

    def clear(self):
        self.positive_region.clear()
        self.negative_region.clear()

    def update(self, X, y):
        """
            Increments the polytope model with new labeled data

            :param X: data matrix
            :param y: labels array. Expects 1 for positive points, and 0 for negative points
        """
        X, y = np.atleast_2d(X), np.asarray(y)

        positive_mask = (y == 1)

        if np.any(positive_mask):
            pos_points = X[positive_mask]
            self.positive_region.update(pos_points)
            self.negative_region.update_positive_region(pos_points)

        negative_mask = ~positive_mask
        if np.any(negative_mask):
            self.negative_region.update(X[negative_mask])

    def predict(self, X):
        """
            Predicts the label of each data point. Returns 1 if point is in positive region, 0 if in negative region,
            and 0.5 otherwise

            :param X: data matrix to predict labels
        """
        X = np.atleast_2d(X)

        probas = np.full(shape=(X.shape[0],), fill_value=0.5)

        if self.positive_region.is_built:
            probas[self.positive_region.is_inside(X)] = 1.0

        if self.negative_region.is_built:
            probas[self.negative_region.is_inside(X)] = 0.0

        return probas


class PositiveRegion:
    def __init__(self, tol):
        assert_positive(tol, 'tol')

        self.hull = None
        self.cache = None
        self.tol = tol

    @property
    def is_built(self):
        return self.hull is not None

    @property
    def has_facet(self):
        return self.is_built or self.cache.shape[0] >= self.cache.shape[1]

    @property
    def vertices(self):
        return self.cache if self.hull is None else self.hull.vertices

    def clear(self):
        self.hull = None
        self.cache = None

    def is_inside(self, X):
        if not self.hull:
            return np.full(len(X), False)

        return self.hull.is_inside(X)

    def update(self, pos_points):
        if self.is_built:
            self.hull.add_points(pos_points)
            return

        self.__update_cache(pos_points)

        if self.cache.shape[0] > self.cache.shape[1]:
            self.hull = self.__build_convex_hull()
            self.cache = None

    def __update_cache(self, X):
        if self.cache is None:
            self.cache = X.copy()
        else:
            self.cache = np.vstack([self.cache, X])

    def __build_convex_hull(self):
        return ConvexHull(self.cache, self.tol) if self.cache.shape[1] > 1 else OneDimensionalConvexHull(self.cache)


class NegativeRegion:
    def __init__(self, positive_region, tol):
        assert_positive(tol, 'tol')

        self.positive_region = positive_region
        self.cones = []
        self.cache = None
        self.tol = tol

    @property
    def is_built(self):
        return len(self.cones) > 0

    def clear(self):
        self.positive_region.clear()
        self.cones = []
        self.cache = None

    def is_inside(self, X):
        if not self.cones:
            return np.full(len(X), False)

        return np.any([nr.is_inside(X) for nr in self.cones], axis=0)

    def update_positive_region(self, pos_points):
        for nr in self.cones:
            nr.add_points_to_hull(pos_points)

        # if positive region has at least a facet, and cache has not been used, build negative cones
        if self.positive_region.has_facet and self.cache is not None:
            vertices = self.positive_region.vertices

            for neg_point in self.cache:
                self.cones.append(self.__build_convex_cone(vertices, neg_point))

            self.cache = None

    def update(self, neg_points):
        if self.positive_region.has_facet:
            vertices = self.positive_region.vertices

            for neg_point in neg_points:
                self.cones.append(self.__build_convex_cone(vertices, neg_point))

            return

        self.__update_cache(neg_points)

    def __build_convex_cone(self, vertices, neg_point):
        return ConvexCone(vertices, neg_point, self.tol) if len(neg_point) > 1 else OneDimensionalConvexCone(vertices, neg_point)

    def __update_cache(self, X):
        if self.cache is None:
            self.cache = X.copy()
        else:
            self.cache = np.vstack([self.cache, X])
