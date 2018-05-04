from .kernel import KernelLogisticRegression
from .linear import BayesianLogisticRegression
from ..uncertainty import UncertaintySampler


class LinearQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0, rounding=True):
        clf = BayesianLogisticRegression(n_samples, add_intercept, sampling, warmup, thin, sigma, rounding)
        UncertaintySampler.__init__(self, clf)


class KernelQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0, rounding=True,
                       kernel='linear', gamma=None, degree=3, coef0=0.):
        clf = KernelLogisticRegression(n_samples, add_intercept, sampling, warmup, thin, sigma, rounding, kernel, gamma, degree, coef0)
        UncertaintySampler.__init__(self, clf)
