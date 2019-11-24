import numpy as np
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted

from .sampling import StanLogisticRegressionSampler, HitAndRunSampler


class BayesianLogisticRegression:
    """
        Logistic regression bayesian classifier. Given a collection of labeled data (X_i, y_i), we assume two choices of posterior:

        LOGISTIC POSTERIOR
            p(w | X, y) ~= exp(-|w|^2 / sigma^2) * \prod_i 1 / (1 + exp(-y_i X_i^T w))

        Basically, we have chosen a centered gaussian prior and a logistic likelihood function.

        DETERMINISTIC POSTERIOR
            p(w | X, y) = 1 if y_i X_i^T w > 0 for all i, else 0

        In this particular case, there we assume no labeling noise, and data should be linear separable. However, we can achieve
        better performance under these assumptions.
    """

    def __init__(self, n_samples, add_intercept=True, sampling='deterministic', warmup=100, thin=1, sigma=100, rounding=True):
        """
        :param n_samples: number of samples to compute from posterior
        :param add_intercept: whether to add an intercept or not
        :param sampling: sampling method. Options: 'bayesian' (allows labeling noise) and 'deterministic' (no noise)
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param sigma: gaussian prior standard deviation. Works as a L2 regularization (the lower sigma is, the more regularization)
        :param rounding: whether to apply a rounding procedure in the 'deterministic' sampling.
        """
        if sampling == 'bayesian':
            self.sampler = StanLogisticRegressionSampler(warmup=warmup, thin=thin, sigma=sigma)
        elif sampling == 'deterministic':
            self.sampler = HitAndRunSampler(warmup=warmup, thin=thin, rounding=rounding, cache=True)
        else:
            raise ValueError("Unknown sampling option. Options are 'deterministic' and 'bayesian'.")

        self.n_samples = n_samples
        self.add_intercept = add_intercept
        self.sampling = sampling

    def fit(self, X, y):
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

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X):
        check_is_fitted(self, ('weight', 'bias'))

        return np.mean(self.__likelihood(X), axis=0)

    def __likelihood(self, X):
        margin = self.__margin(X)
        if self.sampling == 'bayesian':
            return expit(margin)
        else:
            return (margin > 0).astype('float')

    def __margin(self, X):
        return self.bias + self.weight.dot(X.T)
