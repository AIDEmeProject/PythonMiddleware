import numpy as np
from scipy.special import expit
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from .sampling import StanLogisticRegressionSampler, HitAndRunSampler


class BayesianLogisticRegression:
    """
        Logistic regression bayesian model.
    """

    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0):
        if sampling == 'bayesian':
            self.sampler = StanLogisticRegressionSampler(warmup, thin, sigma)
        elif sampling == 'deterministic':
            self.sampler = HitAndRunSampler(warmup, thin)
        else:
            raise ValueError("Unknown sampling backend. Options are 'stan' or 'hit-and-run'.")

        self.sampling = sampling
        self.n_samples = n_samples
        self.add_intercept = add_intercept

    def fit(self, X, y):
        # check data
        X, y = check_X_y(X, y)

        # add intercept if needed
        if self.add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])

        # call sampling function
        samples = self.sampler.sample(X, y, self.n_samples)

        # set bias and weight
        if self.add_intercept:
            self.bias, self.weight = samples[:, 0].reshape(-1, 1), samples[:, 1:]
        else:
            self.bias, self.weight = 0, samples

    def __likelihood(self, X):
        if self.sampling == 'bayesian':
            return expit(X)
        else:
            return (X > 0).astype('float')

    def predict_proba(self, X):
        check_is_fitted(self, ('weight', 'bias'))

        X = check_array(X)

        return np.mean(self.__likelihood(self.bias + self.weight.dot(X.T)), axis=0)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('float')
