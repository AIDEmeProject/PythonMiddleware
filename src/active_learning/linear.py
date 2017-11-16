from .base import ActiveLearner
from ..convexbody.sampling import HitAndRunSampler
from ..version_space.svm import SVMVersionSpace
import numpy as np


class LinearMajorityVote(ActiveLearner):
    def __init__(self, chain_length=100, estimate_cut=False):
        super().__init__(estimate_cut)
        self.sampler = HitAndRunSampler(chain_length)
        self.samples = None

    def initialize(self, data):
        self.version_space = SVMVersionSpace(data.shape[1])

    def fit_classifier(self, X, y):
        pass

    def get_majority_vote(self, X):
        predictions = np.sign(np.dot(X, self.samples.T))
        return np.sum(predictions, axis=-1)

    def predict(self, X):
        vote = self.get_majority_vote(X)
        return 2.*(vote >= 0) - 1.

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        return self.sampler.sample_chain(self.version_space, initial_point)

    def update(self, points, labels):
        super().update(points, labels)
        self.samples = self.sample_direction()

    def ranker(self, data):
        vote = self.get_majority_vote(data)
        return np.abs(vote)

