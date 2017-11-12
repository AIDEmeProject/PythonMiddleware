from .base import ActiveLearner
from ..convexbody.sampling import HitAndRunSampler
from ..version_space.svm import SVMVersionSpace
import numpy as np


class LinearMajorityVote(ActiveLearner):
    def __init__(self, chain_length=100):
        super().__init__()
        self.sampler = HitAndRunSampler(chain_length)
        self.samples = None

    def initialize(self, data):
        self.version_space = SVMVersionSpace(data.shape[1])

    def fit_classifier(self, X, y):
        pass

    def predict(self, X):
        return 2.*(np.sum(np.sign(np.dot(X, self.samples.T)), axis=-1) >= 0) - 1.

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        return self.sampler.sample_chain(self.version_space, initial_point)

    def update(self, points, labels):
        super().update(points, labels)
        self.samples = self.sample_direction()

    def ranker(self, data):
        samples = self.sample_direction()
        prediction = np.sign(data.dot(samples.T))
        return np.abs(np.sum(prediction, axis=-1))
