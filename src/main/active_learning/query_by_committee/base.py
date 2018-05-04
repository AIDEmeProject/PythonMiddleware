from .kernel import KernelLogisticRegression
from .linear import BayesianLogisticRegression
from ..uncertainty import UncertaintySampler


class LinearQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0):
        clf = BayesianLogisticRegression(n_samples, add_intercept, sampling, warmup, thin, sigma)
        UncertaintySampler.__init__(self, clf)


class KernelQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0,
                       kernel='linear', gamma=None, degree=3, coef0=0.):
        clf = KernelLogisticRegression(n_samples, add_intercept, sampling, warmup, thin, sigma, kernel, gamma, degree, coef0)
        UncertaintySampler.__init__(self, clf)
