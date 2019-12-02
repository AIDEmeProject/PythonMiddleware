import numpy as np

from ..dsm.persistent import CategoricalPolytope, MultiSetPolytope
from ..active_learner import ActiveLearner


class PolytopeLearner(ActiveLearner):
    """
    Special AL where predictions are simply based over a polytope model instance.
    """
    def __init__(self, pol):
        self.pol = pol

    def clear(self):
        """
        Clears polytope model
        """
        self.pol.clear()

    def fit_data(self, data):
        """
        Updates the polytope model with the last user labeled data.
        :param data: PartitionedDataset instance
        """
        X_new, y_new = data.last_labeled_set
        self.__update_polytope(X_new, y_new)

    def fit(self, X, y):
        """
        Similar to fit_data, but polytope model is cleared before fitting.
        :param X: data matrix
        :param y: labels
        """
        self.clear()
        self.__update_polytope(X, y)

    def __update_polytope(self, X_new, y_new):
        self.pol.update(X_new, y_new)

        if not self.pol.is_valid:
            raise RuntimeError('Found conflicting labels in polytope: {}'.format(self.pol))

    def predict(self, X):
        """
        Returns the most probable label. 0.5 probabilities are treated as negative labels.
        :param X: data matrix to compute labels
        :return: numpy array containing labels for each data point
        """
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X):
        """
        :param X: data matrix
        :return: polytope predictions for eac data point x in X
        """
        return self.pol.predict(X)

    def rank(self, X):
        """
        :param X: data matrix
        :return: 0 for points in positive or negative regions, 0.5 otherwise
        """
        return np.where(self.predict_proba(X) == 0.5, 0.5, 0)


class CategoricalActiveLearner(PolytopeLearner):
    """
    Special AL for the case where all attributes are categorical. It simply memorizes the positive and negative values
    seen so far.
    """
    def __init__(self):
        super().__init__(CategoricalPolytope())


class MultiSetActiveLearner(ActiveLearner):
    """
    Special AL for the case where all attributes are come from a multi-set feature. It simply memorizes the positive
    values seen so far, and negative points are cached until there is only one element left, which will assumed to be
    negative.
    """
    def __init__(self):
        super().__init__(MultiSetPolytope())
