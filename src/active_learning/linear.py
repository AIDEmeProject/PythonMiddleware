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
        self.__majority_vote = None

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


from sklearn.svm import SVC
class KernelMajorityVote(ActiveLearner):
    def __init__(self, chain_length=50, sample_size=8, top=-1):
        super().__init__(top)
        self.sample_size = sample_size
        self.chain_length = chain_length
        self.__data = None
        self.__K = None
        self.__labeled_indexes = []
        self.__labels = []
        self.clf = SVC(C=1000, kernel='rbf')

    def clear(self):
        self.__data = None
        self.__K = None
        self.__majority_vote = None
        self.__labeled_indexes = []
        self.__labels = []
        self.clf = SVC(C=100000, kernel='rbf')

    def initialize(self, data):
        self.__data = data.values

    def update(self, points, labels):
        # udpate labels and indexes
        indexes = points.index
        self.__labels.extend(labels.values)
        self.__labeled_indexes.extend(indexes)

        # create new version space
        self.__K = rbf_kernel(self.__data[self.__labeled_indexes])
        self.version_space = LinearVersionSpace(len(self.__labeled_indexes))
        for point, label in zip(self.__K, self.__labels):
            self.version_space.update(point, label)

    def ranker(self, data):
        samples = self.version_space.sample(self.chain_length, self.sample_size)
        bias, weight = samples[:, 0].reshape(-1, 1), samples[:, 1:]
        predictions = np.sign(bias + weight.dot(rbf_kernel(data[self.__labeled_indexes], data)))
        return np.abs(np.sum(predictions, axis=0))

