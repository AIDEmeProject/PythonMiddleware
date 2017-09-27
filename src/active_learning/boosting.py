import numpy as np
from .base import ActiveLearner
from ..utils import Adaboost
from ..convexbody.markov_samplers import HitAndRunSampler
from ..convexbody.polytope import ActBoostPolytope


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
        self.chain_length = chain_length
        self.search_space = None
        self.hit_and_run = None

    def initialize(self, data):
        self.search_space = ActBoostPolytope(data.shape[1])
        self.hit_and_run = HitAndRunSampler(self.chain_length, self.search_space)

    def clear(self):
        self.search_space.clear()

    def get_next(self, pool):
        q0 = self.search_space.get_point()
        samples = self.hit_and_run.uniform(q0, self.sample_size)
        return pool.find_minimizer(lambda x: np.abs(np.sum(np.sign(np.dot(x, samples.T)), axis=-1)))

    def update(self, X, y):
        try:
            self.search_space.append(X, float(y))
        except:
            for point, label in zip(X, y):
                self.search_space.append(point, label)



