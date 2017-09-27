import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .datapool import Point


def label_all(data, user):
    points = Point(index=range(len(data)), data=data)
    return user.get_label(points, update_counter=False)


class Adaboost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iterations=300, weights=None, alphas=None):
        """
        Called when initializing the classifier
        """
        self.n_iterations = n_iterations
        self.weights = weights
        self.alphas = alphas

    def fit(self, X, y):
        """
        Xij = hj(xi)
        """
        X = np.array(X)
        y = np.array(y)
        n, k = X.shape

        if set(X.flatten()) != {-1, 1}:
            raise AttributeError("X matrix must be composed of -1 and 1 only!")

        if set(y) != {-1, 1}:
            raise AttributeError("y vector must be composed of -1 and 1 only!")

        self.weights = np.ones(n) / n
        self.alphas = np.zeros(k)
        mask = (X * y.reshape(-1, 1)) == -1

        for t in range(self.n_iterations):
            # compute eps
            eps = np.dot(self.weights, mask)

            # get min eps
            idx_min = np.argmin(eps)
            eps_min = eps[idx_min]

            if eps_min == 0:
                self.alphas[idx_min] = 1
                break

            # update alpha
            alpha_t = 0.5 * np.log((1 - eps_min) / eps_min)
            self.alphas[idx_min] += alpha_t

            # update weights
            self.weights *= np.exp(-alpha_t * y * X[:, idx_min])
            self.weights /= np.sum(self.weights)

        return self

    def predict(self, X, y=None):
        return 2.0*(X.dot(self.alphas) >= 0) - 1.0

    def score(self, X, y):
        # counts number of values bigger than mean
        return np.sum(self.predict(X) == y) / len(y)


