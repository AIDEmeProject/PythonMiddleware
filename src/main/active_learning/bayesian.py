import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel

from .base import ActiveLearner

class LogisticPdf:
    def __init__(self, X, Y, add_bias=True, sig0=1.):
        self.var = sig0 ** 2

        X = np.atleast_2d(X)
        if add_bias:
            X = np.hstack([np.ones(shape=(len(X), 1)), X])

        Y = np.array(Y).ravel()

        # prod_ij = y_i * x_ij
        self.prod = X * Y.reshape(-1, 1)
        self.mask = self.prod > 0

    def __sigmoid(self, th):
        """
        Sigmoid function applied to all data points: sigma(th) = 1 / (1 + exp(- y_i <x_i, th>))

        th: input value
        """
        return expit(self.prod.dot(th))

    def sample_height(self, th):
        """
        Sample an uniform height from [0, f_i(th)]

        th: point to sample height
        """
        return np.random.uniform(low=0, high=self.__sigmoid(th))

    def sample_intersection(self, th, j, heights):
        """
        Sample from the set {th_j: f_i(th_j | th_-j) > height_i, for all i}

        th     : current theta value
        j      : dimension to sample
        heights: sampled heights (do not include prior)
        """

        # bounds_i = log(h_i / (1 - h_i)) - sum_{k != j} prod_ik th_k
        bounds = np.log(heights / (1 - heights)) - self.prod.dot(th) + self.prod[:, j] * th[j]

        # prod_ij th_j >= bounds_i, for all i
        extremes = bounds / self.prod[:, j]
        min_bound = np.max(extremes[self.mask[:, j]]) if np.any(self.mask[:, j]) else -float('inf')
        max_bound = np.min(extremes[~self.mask[:, j]]) if np.any(~self.mask[:, j]) else float('inf')

        # gaussian prior constrains
        limit = np.sqrt(th[j] ** 2 - 2 * self.var * np.log(
            np.random.rand()))  # np.sqrt(np.sum(th**2) - 2*self.var*np.log(np.random.rand()))

        min_bound = max(min_bound, -limit)
        max_bound = min(max_bound, limit)

        return np.random.uniform(min_bound, max_bound)


def slice_sampler(p, th0, N, burn_in=0, skip=1):
    th = np.array(th0, copy=True, dtype='float').ravel()
    dim = len(th)

    samples = []
    count = 0

    for _ in range(N):
        for j in range(dim):
            # sample uniformly from [0, f_i(th)] for each i
            heights = p.sample_height(th)

            # sample th[j] uniformly from {th_j: p(th_j | th_-j) >= height}
            th[j] = p.sample_intersection(th, j, heights)

        # update samples
        if count >= burn_in and (count - burn_in) % skip == 0:
            samples.append(th.copy())
        count += 1

    return np.array(samples)


class BayesianLogisticActiveLearner(ActiveLearner):
    def __init__(self, add_bias=True, n_samples=1000, burn_in=-1, skip=1):
        super().__init__()
        self.add_bias = add_bias
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.skip = skip
        self.samples = None

    def clear(self):
        self.samples = None

    def predict(self, X):
        return 2 * (self.predict_proba(X) > 0.5) - 1

    def predict_proba(self, X):
        bias, weights = self.samples[:, 0].reshape(-1, 1), self.samples[:, 1:]
        return np.mean(expit(bias + weights.dot(np.transpose(X))), axis=0)

    def fit_classifier(self, X, y):
        p = LogisticPdf(X, y, self.add_bias, 1.0)
        self.samples = slice_sampler(p, np.zeros(X.shape[1] + self.add_bias), self.n_samples, self.burn_in, self.skip)

    def ranker(self, data):
        probas = self.predict_proba(data)
        return np.abs(probas - 0.5)


class KernelBayesianActiveLearner(ActiveLearner):
    def __init__(self, add_bias=True, n_samples=1000, burn_in=-1, skip=1):
        super().__init__()
        self.linear_learner = BayesianLogisticActiveLearner(add_bias, n_samples, burn_in, skip)
        self.X = None

    def clear(self):
        self.X = None
        self.linear_learner.clear()

    def __preprocess(self, X):
        return rbf_kernel(X, self.X)

    def predict(self, X):
        return self.linear_learner.predict(self.__preprocess(X))

    def predict_proba(self, X):
        return self.linear_learner.predict_proba(self.__preprocess(X))

    def fit_classifier(self, X, y):
        self.X = X.copy()
        self.linear_learner.fit_classifier(self.__preprocess(X), y)

    def ranker(self, data):
        return self.linear_learner.ranker(self.__preprocess(data))