import numpy as np

from ..active_learner import ActiveLearner


class CategoricalActiveLearner(ActiveLearner):
    """
    Special AL for the case where all attributes are assumed to be categorical. It simply memorizes the positive and
    negative values seen so far.
    """
    def __init__(self):
        self._pos_classes = set()
        self._neg_classes = set()

    def clear(self):
        self._pos_classes = set()
        self._neg_classes = set()

    def predict(self, X):
        """
        :return: 1.0 for x in positive set, 0.0 otherwise
        """
        return self.__apply_function_over_data(X, self.__predict_single)

    def __predict_single(self, x):
        return 1.0 if tuple(x) in self._pos_classes else 0.0

    def predict_proba(self, X):
        """
        :return: 1.0 for x in positive set, 0.0 for x in negative set, 0.5 otherwise
        """
        return self.__apply_function_over_data(X, self.__predict_proba_single)

    def __predict_proba_single(self, x):
        x = tuple(x)
        return 1.0 if x in self._pos_classes else 0.0 if x in self._neg_classes else 0.5

    def rank(self, X):
        """
        :return: 0.0 for x in positive or negative sets, 0.5 otherwise
        """
        return self.__apply_function_over_data(X, self.__rank_single)

    def __rank_single(self, x):
        x = tuple(x)
        return 0.0 if (x in self._pos_classes or x in self._neg_classes) else 0.5

    def __apply_function_over_data(self, X, function):
        return np.fromiter((function(x) for x in X), np.float)

    def fit(self, X, y):
        """
        Fits the categorical classifier over the labeled data. It simply memorizes the positive and negative values seen
        so far.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        """
        self.clear()  # TODO: can we avoid calling clear?

        for pt, lb in zip(X, y):
            pt = tuple(pt)  # convert to tuple because numpy arrays are not hashable

            if lb == 1:
                self._pos_classes.add(pt)
            else:
                self._neg_classes.add(pt)

        if len(self._pos_classes & self._neg_classes) > 0:
            raise ValueError("Found conflicting labels for categories {0}".format(self._pos_classes & self._neg_classes))
