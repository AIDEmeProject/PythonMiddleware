import numpy as np

from .base import ActiveLearner
from ..version_space.linear import LinearVersionSpace


class LinearMajorityVote(ActiveLearner):
    def __init__(self, chain_length=50, sample_size=8, top=-1):
        super().__init__(top)
        self.sample_size = sample_size
        self.chain_length = chain_length
        self.samples = None

    def initialize(self, data):
        self.version_space = LinearVersionSpace(data.shape[1])

    def fit_classifier(self, X, y):
        pass

    def get_majority_vote(self, X):
        bias, weight = self.samples[:,0].reshape(-1,1), self.samples[:,1:]
        predictions = np.sign(bias + weight.dot(X.T))
        return np.sum(predictions, axis=0)

    def predict(self, X):
        vote = self.get_majority_vote(X)
        return 2.*(vote >= 0) - 1.

    def update(self, points, labels):
        super().update(points, labels)
        self.samples = self.version_space.sample(self.chain_length, self.sample_size)

    def ranker(self, data):
        vote = self.get_majority_vote(data)
        return np.abs(vote)
