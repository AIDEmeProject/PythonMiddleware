import numpy as np

from ..version_space.boosting import ActboostPolytope
from ..convexbody.sampling.markov_samplers import HitAndRunSampler
from .base import ActiveLearner
from ..utils import Adaboost


class BaseBoosting(ActiveLearner):
    def __init__(self, n_iterations=300):
        super().__init__()
        self.clf = Adaboost(n_iterations=n_iterations)
        

class QueryByBoosting(BaseBoosting):
    def get_next(self, pool):
        return pool.find_minimizer(lambda x: np.abs(np.dot(x, self.clf.alphas)))


class ActBoost(BaseBoosting):
    def __init__(self, sample_size, chain_length, n_iterations=300):
        super().__init__(n_iterations)
        self.sample_size = sample_size
        self.hit_and_run = HitAndRunSampler(chain_length)
        self.version_space = None

    def initialize(self, data):
        self.version_space = ActboostPolytope(data.shape[1])

    def get_next(self, pool):
        # find point inside current version space
        initial_point = self.version_space.get_point()

        # sample uniformly over the current version space
        samples = self.hit_and_run.uniform(self.version_space, initial_point, self.sample_size)

        # find minimizer of current unlabeled pool
        def f(X, samples):
            prediction = 2 * (np.dot(X, samples.T) >= 0) - 1
            return np.abs(np.sum(prediction, axis=-1))

        return pool.find_minimizer(lambda x, s=samples: f(x, s))
