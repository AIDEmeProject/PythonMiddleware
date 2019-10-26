import numpy as np

from .polytope import PolytopeModel


class FactorizedPolytopeModel:
    def __init__(self, partition, tol=1e-12):
        self.partition = partition
        self.polytope_models = [PolytopeModel(tol) for _ in partition]

    def clear(self):
        for pol in self.polytope_models:
            pol.clear()

    def update(self, X, y):
        """
            Updates the polytope models in each subspace

            :param X: data matrix
            :param y: partial labels matrix. Expects 1 for positive labels, and 0 for negative labels
        """
        for i, (idx, pol) in enumerate(zip(self.partition, self.polytope_models)):
            X_subspace, y_subspace = X[:, idx], y[:, i]
            pol.update(X_subspace, y_subspace)

    def predict(self, X):
        """
            Predicts the label of each data point. Returns 1 if point is in positive region, 0 if in negative region,
            and -1 otherwise

            :param X: data matrix to predict labels
        """
        proba = self.predict_proba(X)
        return np.where(proba != 0.5, proba, -1)

    def predict_proba(self, X):
        """
            Predicts the label probabilities of each data point. Returns 1 if point is in positive region, 0 if in negative
            region, and 0.5 otherwise

            :param X: data matrix to predict labels
        """
        prediction = self.polytope_models[0].predict_proba(X[:, self.partition[0]])
        for i in range(1, len(self.partition)):
            np.minimum(prediction, self.polytope_models[i].predict_proba(X[:, self.partition[i]]), out=prediction)

        return prediction
