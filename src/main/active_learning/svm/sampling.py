import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from src.main.active_learning.base import ActiveLearner
from src.main.version_space import LinearVersionSpace


class SamplingBase(ActiveLearner):
    def __init__(self, top=-1, chain_length=50, sample_size=8, cholesky=False):
        super().__init__(top)

        self.sample_size = sample_size
        self.chain_length = chain_length
        self._samples = None

        self.cholesky = cholesky
        self._L = None

        self._data = None
        self._labeled_indexes = []
        self._labels = []

    def clear(self):
        self._data = None
        self._labeled_indexes = []
        self._labels = []
        self._L = None
        self._samples = None

    def initialize(self, data):
        self._data = data.values

    def update(self, points, labels):
        # update labels and indexes
        self._labels.extend(labels.values)
        self._labeled_indexes.extend(points.index)

        # compute kernel matrix
        K = self.get_kernel_matrix(self._data[self._labeled_indexes])
        if self.cholesky:
            K = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
            self._L = K

        # create version space
        self.version_space = LinearVersionSpace(K.shape[1])
        self.version_space.update(K, self._labels)

        self._samples = self.version_space.sample(self.chain_length, self.sample_size)

    def get_kernel_matrix(self, X, Y=None):
        return rbf_kernel(X, Y)

    def ranker(self, data):
        bias, weight = self.get_bias_and_weight()

        K = self.get_kernel_matrix(data, self._data[self._labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return np.abs(np.sum(predictions, axis=0))

    def get_bias_and_weight(self):
        bias, weight = self._samples[:, 0].reshape(-1, 1), self._samples[:, 1:]

        if self.cholesky:
            weight = weight.dot(np.linalg.inv(self._L))
        return bias, weight


class MajorityVote(SamplingBase):
    def fit_classifier(self, X, y):
        pass

    def predict(self, X):
        bias, weight = self.get_bias_and_weight()

        K = self.get_kernel_matrix(X, self._data[self._labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return 2*(np.sum(predictions, axis=0) > 0) - 1