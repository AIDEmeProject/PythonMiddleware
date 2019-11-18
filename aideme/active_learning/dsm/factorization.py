import numpy as np

from .persistent import PersistentPolytopeModel


class FactorizedPolytopeModel:
    def __init__(self, partition, is_convex_positive_list, tol=1e-12):
        if is_convex_positive_list is None or isinstance(is_convex_positive_list, bool):
            is_convex_positive_list = [is_convex_positive_list] * len(partition)

        if len(partition) != len(is_convex_positive_list):
            raise ValueError("Lists have incompatible sizes: {0} and {1}".format(len(partition), len(is_convex_positive_list)))

        self.partition = partition
        self.polytope_models = [PersistentPolytopeModel(flag, tol) for flag in is_convex_positive_list]

    @property
    def is_valid(self):
        return any((pol.is_valid for pol in self.polytope_models))

    def clear(self):
        for pol in self.polytope_models:
            pol.clear()

    def update(self, X, y):
        """
            Updates the polytope models in each subspace

            :param X: data matrix
            :param y: partial labels matrix. Expects 1 for positive labels, and 0 for negative labels
        """
        if not self.is_valid:
            raise RuntimeError("Cannot update invalid polytope.")

        return all(pol.update(X[:, idx], y[:, i]) for i, idx, pol in self.__valid_elements())

    def predict(self, X):
        """
            Predicts the label of each data point. Returns 1 if point is in positive region, 0 if in negative region,
            and -1 otherwise

            :param X: data matrix to predict labels
        """
        if not self.is_valid:
            return np.full(len(X), fill_value=0.5)

        prediction = np.full(len(X), fill_value=1.0)

        for i, idx, pol in self.__valid_elements():
            np.minimum(prediction, pol.predict(X[:, idx]), out=prediction)

        return prediction

    def predict_partial(self, X):
        partial_labels = np.full((X.shape[0], len(self.partition)), fill_value=0.5)

        for i, idx, pol in self.__valid_elements():
            partial_labels[:, i] = pol.predict(X[:, idx])

        return partial_labels

    def __valid_elements(self):
        return ((i, idx, pol) for i, (idx, pol) in enumerate(zip(self.partition, self.polytope_models)) if pol.is_valid)