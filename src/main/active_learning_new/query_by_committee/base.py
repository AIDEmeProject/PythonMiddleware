from .sampler import StanLogisticRegressionSampler
from .linear import BayesianLogisticRegression
from .kernel import KernelBayesianLogisticRegression
from ..uncertainty import UncertaintySampler


class LinearQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, warmup=100, thin=1, sigma=1.0):
        sampler = StanLogisticRegressionSampler(warmup, thin, sigma)
        clf = BayesianLogisticRegression(sampler, n_samples, add_intercept)
        UncertaintySampler.__init__(self, clf)


class KernelQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, warmup=100, thin=1, sigma=1.0,
                 kernel='linear', gamma=None, degree=3, coef0=0.):
        sampler = StanLogisticRegressionSampler(warmup, thin, sigma)
        clf = KernelBayesianLogisticRegression(sampler, n_samples, add_intercept, kernel, gamma, degree, coef0)
        UncertaintySampler.__init__(self, clf)
