import numpy as np

from .polytope import PolytopeModel


class PersistentPolytopeModel:
    def __init__(self, is_positive_convex=None, tol=1e-12):
        """
        :param is_positive_convex: flag telling where to build polytope model. There are three possible cases:
                1) True: assume positive region is convex
                2) False: assume negative region is convex
                3) None: run both options above in parallel until one of the polytopes becomes invalid.

        :param tol: polytope model tolerance
        """
        self._pol = self.__get_polytope(is_positive_convex, tol)

    def __get_polytope(self, is_positive_convex, tol):
        if is_positive_convex is None:
            return PersistentPolytope(tol)

        if is_positive_convex:
            return PolytopeModel(tol)

        return FlippedPolytope(tol)

    @property
    def is_valid(self):
        return self._pol.is_valid

    def clear(self):
        self._pol.clear()

    def predict(self, X):
        """
            Predicts the label of each data point. There are three cases to consider:
                1) is_positive_convex == None: if a single polytope is valid, return its predictions. Otherwise, predict all as unknown
                2) is_positive_convex == True: return prediction of polytope built over positive region
                3) is_positive_convex == False: return prediction of polytope built over negative region

            :param X: data matrix to predict labels
        """
        return self._pol.predict(X)


    def update(self, X, y):
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        return self._pol.update(X, y)


class FlippedPolytope:
    """
    Polytope model built assuming the negative region is convex.
    """
    def __init__(self, tol=1e-12):
        """
        :param tol: polytope model tolerance
        """
        self._pol = PolytopeModel(tol)

    @property
    def is_valid(self):
        return self._pol.is_valid

    def clear(self):
        self._pol.clear()

    def predict(self, X):
        """
        :param X: data matrix to predict labels
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        return 1 - self._pol.predict(X)

    def update(self, X, y):
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        return self._pol.update(X, 1 - y)


class PersistentPolytope:
    """
    Polytope model built assuming the negative region is convex.
    """
    def __init__(self, tol=1e-12):
        """
        :param tol: polytope model tolerance
        """
        self._pol = PolytopeModel(tol)
        self._flipped = FlippedPolytope(tol)

    @property
    def is_valid(self):
        return self._pol.is_valid or self._flipped.is_valid

    def clear(self):
        self._pol.clear()
        self._flipped.clear()

    def predict(self, X):
        """
        :param X: data matrix to predict labels
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        X = np.atleast_2d(X)

        if self._pol.is_valid == self._flipped.is_valid:
            return np.full(len(X), fill_value=0.5)

        if self._pol.is_valid:
            return self._pol.predict(X)

        return self._flipped.predict(X)

    def update(self, X, y):
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        if self._pol.is_valid and self._flipped.is_valid:
            return self._pol.update(X, y) and self._flipped.update(X, y)

        if self._pol.is_valid:
            return self._pol.update(X, y)

        return self._flipped.update(X, y)
