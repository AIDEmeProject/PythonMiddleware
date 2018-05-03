import numpy as np
from scipy.special import expit
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class BayesianLogisticRegression:
    """
    Logistic regression bayesian model.
    """

    def __init__(self, sampler, n_samples, add_intercept=True):
        self.sampler = sampler
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

    def predict_proba(self, X):
        check_is_fitted(self, ('weight', 'bias'))

        X = check_array(X)

        return np.mean(expit(self.bias + self.weight.dot(X.T)), axis=0)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('float')
