from .kernel import KernelLogisticRegression
from .linear import BayesianLogisticRegression
from ..uncertainty import UncertaintySampler


class LinearQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0, rounding=True):
        clf = BayesianLogisticRegression(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling,
                                         warmup=warmup, thin=thin, sigma=sigma, rounding=rounding)
        UncertaintySampler.__init__(self, clf)


class KernelQueryByCommittee(UncertaintySampler):
    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0, rounding=True,
                       kernel='linear', gamma=None, degree=3, coef0=0.):
        clf = KernelLogisticRegression(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling,
                                       warmup=warmup, thin=thin, sigma=sigma, rounding=rounding,
                                       kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        UncertaintySampler.__init__(self, clf)
