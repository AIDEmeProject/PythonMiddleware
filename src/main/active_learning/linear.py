from sklearn.metrics.pairwise import rbf_kernel
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


class KernelMajorityVote(ActiveLearner):
    def __init__(self, chain_length=50, sample_size=8, top=-1):
        super().__init__(top)
        self.sample_size = sample_size
        self.chain_length = chain_length
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []
        self.__samples = None

    def clear(self):
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []
        self.__samples = None

    def initialize(self, data):
        self.__data = data.values

    def get_kernel_matrix(self, X, Y=None):
        return rbf_kernel(X, Y)

    def update(self, points, labels):
        # update labels and indexes
        self.__labels.extend(labels.values)
        self.__labeled_indexes.extend(points.index)

        # create new version space
        K = self.get_kernel_matrix(self.__data[self.__labeled_indexes])
        self.version_space = LinearVersionSpace(K.shape[1])
        for point, label in zip(K, self.__labels):
            self.version_space.update(point, label)

        self.__samples = self.version_space.sample(self.chain_length, self.sample_size)

    def ranker(self, data):
        bias, weight = self.__samples[:, 0].reshape(-1, 1), self.__samples[:, 1:]

        K = self.get_kernel_matrix(data, self.__data[self.__labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return np.abs(np.sum(predictions, axis=0))

    def fit_classifier(self, X, y):
        pass

    def predict(self, X):
        bias, weight = self.__samples[:, 0].reshape(-1, 1), self.__samples[:, 1:]

        K = self.get_kernel_matrix(X, self.__data[self.__labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return 2*(np.sum(predictions, axis=0) > 0) - 1


class KernelCholeskyMajorityVote(ActiveLearner):
    def __init__(self, chain_length=50, sample_size=8, top=-1):
        super().__init__(top)
        self.sample_size = sample_size
        self.chain_length = chain_length
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []
        self.__samples = None
        self.L = None

    def clear(self):
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []
        self.__samples = None
        self.L = None

    def initialize(self, data):
        self.__data = data.values

    def get_kernel_matrix(self, X, Y=None):
        return rbf_kernel(X, Y)

    def update(self, points, labels):
        # update labels and indexes
        self.__labels.extend(labels.values)
        self.__labeled_indexes.extend(points.index)

        # create new version space
        K = self.get_kernel_matrix(self.__data[self.__labeled_indexes])
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        self.version_space = LinearVersionSpace(self.L.shape[1])
        for point, label in zip(self.L, self.__labels):
            self.version_space.update(point, label)

        self.__samples = self.version_space.sample(self.chain_length, self.sample_size)

    def ranker(self, data):
        bias, weight = self.__samples[:, 0].reshape(-1, 1), self.__samples[:, 1:]

        K = self.get_kernel_matrix(data, self.__data[self.__labeled_indexes])
        weight = weight.dot(np.linalg.inv(self.L))
        predictions = np.sign(bias + weight.dot(K.T))
        return np.abs(np.sum(predictions, axis=0))

    def fit_classifier(self, X, y):
        pass

    def predict(self, X):
        bias, weight = self.__samples[:, 0].reshape(-1, 1), self.__samples[:, 1:]
        weight = weight.dot(np.linalg.inv(self.L))
        K = self.get_kernel_matrix(X, self.__data[self.__labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return 2*(np.sum(predictions, axis=0) > 0) - 1