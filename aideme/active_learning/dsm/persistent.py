import numpy as np

from .polytope import Polytope


class PolytopeModel:
    def __init__(self, mode='persist', tol=1e-12):
        """
        :param mode: flag specifying the type of polytope to build. There are four possible cases:
                1) 'positive': assume positive region is convex
                2) 'negative': assume negative region is convex
                3) 'persist': run both options above in parallel until one of the polytopes becomes invalid.
                4) 'categorical': special polytope for the case where all attributes are categorical.
                5) 'multiset': special polytope for the case where attributes come from a multi-set encoding.

        :param tol: polytope model tolerance
        """
        self._pol = self.__get_polytope(mode, tol)

    @staticmethod
    def __get_polytope(mode, tol):
        if mode == 'positive':
            return Polytope(tol)
        if mode == 'negative':
            return FlippedPolytope(tol)
        if mode == 'persist':
            return PersistentPolytope(tol)
        if mode == 'categorical':
            return CategoricalPolytope()
        if mode == 'multiset':
            return MultiSetPolytope()
        raise ValueError('Unknown mode {0}. Available values are: {1}'.format(mode, ['categorical', 'multiset', 'negative', 'persist', 'positive']))

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
        self._pol = Polytope(tol)

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
        self._pol = Polytope(tol)
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


class CategoricalPolytope:
    """
    Special polytope for the case where all attributes are assumed to be categorical. It simply memorizes the positive and
    negative values seen so far.
    """
    def __init__(self):
        self._pos_classes = set()
        self._neg_classes = set()
        self.__is_valid = True

    @property
    def is_valid(self):
        return self.__is_valid

    def clear(self):
        self._pos_classes = set()
        self._neg_classes = set()
        self.__is_valid = True

    def predict(self, X):
        """
        :param X: data array to predict labels.
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        if not self.__is_valid:
            return np.full(len(X), fill_value=0.5)

        return np.fromiter((self.__predict_single(x) for x in X), np.float)

    def __predict_single(self, x):
        x = tuple(x)

        if x in self._pos_classes:
            return 1.0

        if x in self._neg_classes:
            return 0.0

        return 0.5

    def update(self, X, y):
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        for pt, lb in zip(X, y):
            pt = tuple(pt)  # convert to tuple because numpy arrays are not hashable

            if lb == 1:
                self._pos_classes.add(pt)
            else:
                self._neg_classes.add(pt)

        self.__is_valid = len(self._pos_classes & self._neg_classes) == 0

        return self.__is_valid


class MultiSetPolytope:
    def __init__(self):
        self._pos_indexes = set()
        self._neg_indexes = set()
        self._neg_point_cache = []
        self.__is_valid = True

    @property
    def is_valid(self):
        return self.__is_valid

    def clear(self):
        self._pos_indexes = set()
        self._neg_indexes = set()
        self._neg_point_cache = []
        self.__is_valid = True

    def predict(self, X):
        """
        :param X: data array to predict labels.
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        pred = np.full(len(X), fill_value=0.5)

        if not self.__is_valid:
            return pred

        if self._pos_indexes:
            pos = list(self._pos_indexes)
            pred[X[:, pos].sum(axis=1) == X.sum(axis=1)] = 1.0

        if self._neg_indexes:
            neg = list(self._neg_indexes)
            pred[X[:, neg].any(axis=1)] = 0.0

        return pred

    def update(self, X, y):
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix. We expect it to be a 0-1 encoding of the multi-set.
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        for pt, lb in zip(X, y):
            if not self.__update_single(pt, lb):
                self.__is_valid = False
                return False

        return True

    def __update_single(self, pt, lb):
        """
        Increments polytope with a single point.
        :param pt: data point
        :param lb: data point label
        :return: whether update was successful or not
        """
        idx = {i for i, ind in enumerate(pt) if ind > 0}

        if lb == 1:
            return self.__update_positive_set(idx)

        idx -= self._pos_indexes
        if len(idx) == 1:
            return self.__update_negative_set(idx)

        self._neg_point_cache.append(idx)
        return True

    def __update_positive_set(self, idx):
        """
        Adds new indexes to the positive set. Also, removes the new positive indexes from each negative point in the cache.

        :param idx: set of new positive indexes
        """
        if not idx:
            return True

        if len(idx & self._neg_indexes) > 0:
            return False

        self._pos_indexes.update(idx)

        new_neg = set()

        for cache in self._neg_point_cache:
            cache.difference_update(idx)

            if len(cache) == 0:
                return False

            elif len(cache) == 1:
                new_neg.update(cache)

        return self.__update_negative_set(new_neg)

    def __update_negative_set(self, idx):
        """
        Adds new indexes to the negative set. Also, if a cached negative points contains any of the new negative indexes,
        it will also be removed from cache since no inference can be done on its values.

        :param idx: set of new negative indexes
        """
        if not idx:
            return True

        if len(idx & self._pos_indexes) > 0:
            return False

        self._neg_indexes.update(idx)
        self._neg_point_cache = [neg_idx for neg_idx in self._neg_point_cache if len(idx & neg_idx) == 0]

        return True