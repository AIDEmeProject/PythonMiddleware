import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel

from .base import ActiveLearner


class LogisticPdf:
    def __init__(self, X, Y, add_bias=True, sig0=1., class_weight=True):
        X = np.array(X, dtype='float')
        X = X.reshape(len(X), -1)
        if add_bias:
            X = np.hstack([np.ones(shape=(len(X), 1)), X])

        Y = np.array(Y, dtype='float')
        Y = Y.reshape(-1, 1)
        if class_weight:
            pos, neg = len(Y) / (2 * sum(Y == 1)), -len(Y) / (2 * sum(Y == -1))
            Y = np.where(Y == 1, pos, neg)

        # prod_ij = y_i * x_ij
        self.prod = Y * X
        self.pos_mask = self.prod > 0
        self.neg_mask = self.prod < 0

        # variance of gaussian prior
        self.var = float(sig0) ** 2

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
        min_bound = np.max(extremes[self.pos_mask[:, j]]) if np.any(self.pos_mask[:, j]) else -float('inf')
        max_bound = np.min(extremes[self.neg_mask[:, j]]) if np.any(self.neg_mask[:, j]) else float('inf')

        # gaussian prior constrains
        limit = np.sqrt(th[j] ** 2 - 2 * self.var * np.log(np.random.rand()))  # np.sqrt(np.sum(th**2) - 2*self.var*np.log(np.random.rand()))
        min_bound = max(min_bound, -limit)
        max_bound = min(max_bound, limit)

        return np.random.uniform(min_bound, max_bound)


class BayesianLogisticActiveLearner(ActiveLearner):
    def __init__(self, add_bias=True, n_samples=1000, burn_in=-1,
                 skip=1, sig0=1., class_weight=True):
        super().__init__()

        self.add_bias     = bool(add_bias)
        self.n_samples    = int(n_samples)
        self.burn_in      = int(burn_in)
        self.skip         = int(skip)
        self.sig0         = float(sig0)
        self.class_weight = bool(class_weight)
        self.samples      = None

    def clear(self):
        self.samples = None

    def predict(self, X):
        return 2 * (self.predict_proba(X) > 0.5) - 1

    def predict_proba(self, X):
        if self.add_bias:
            bias, weights = self.samples[:, 0].reshape(-1, 1), self.samples[:, 1:]
        else:
            bias, weights = 0, self.samples
        return np.mean(expit(bias + weights.dot(np.transpose(X))), axis=0)

    def fit_classifier(self, X, y):
        p = LogisticPdf(X, y, self.add_bias, self.sig0, self.class_weight)
        start = np.zeros(X.shape[1] + self.add_bias)
        if self.samples is not None:
            start[:self.samples.shape[1]] = np.mean(self.samples, axis=0)
        self.samples = slice_sampler(p, start, self.n_samples, self.burn_in, self.skip)

    def ranker(self, data):
        return (self.predict_proba(data) - 0.5)**2


class KernelBayesianActiveLearner(ActiveLearner):
    def __init__(self, add_bias=True, n_samples=1000, burn_in=-1, skip=1, sig0=1., class_weight=True, gamma=None):
        super().__init__()

        self.gamma          = gamma
        self.linear_learner = BayesianLogisticActiveLearner(add_bias, n_samples, burn_in,
                                                            skip, sig0, class_weight)
        self.X = None


    def clear(self):
        self.X = None
        self.linear_learner.clear()

    def __preprocess(self, X):
        return rbf_kernel(X, self.X, gamma=self.gamma)

    def predict(self, X):
        return self.linear_learner.predict(self.__preprocess(X))

    def predict_proba(self, X):
        return self.linear_learner.predict_proba(self.__preprocess(X))

    def fit_classifier(self, X, y):
        self.X = X.copy()
        self.linear_learner.fit_classifier(self.__preprocess(X), y)

    def ranker(self, data):
        return self.linear_learner.ranker(self.__preprocess(data))


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
