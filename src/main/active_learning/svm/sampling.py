import numpy as np
from sklearn.svm import SVC

from src.main.active_learning.base import ActiveLearner
from src.main.version_space import KernelVersionSpace


class SamplingBase(ActiveLearner):
    def __init__(self, top=-1, chain_length=50, sample_size=8, kernel='rbf', cholesky=False, rounding=False):
        super().__init__(top)

        self.cholesky = cholesky
        self.kernel = kernel
        self.version_space = KernelVersionSpace(kernel, cholesky, rounding)

        self.sample_size = sample_size
        self.chain_length = chain_length
        self._svm_classifier_sample = None

    def clear(self):
        super().clear()
        self._svm_classifier_sample = None

    def update(self, points, labels):
        self.version_space.update(points, labels)
        self._svm_classifier_sample = self.version_space.sample_classifier(self.chain_length, self.sample_size,
                                                                           self._svm_classifier_sample)
    def ranker(self, data):
        predictions = self._svm_classifier_sample.predict(data)
        return np.abs(np.sum(predictions, axis=0))


class MajorityVote(SamplingBase):
    def fit_classifier(self, X, y):
        self.version_space.clear()
        self.version_space.update(X, y)
        self._svm_classifier_sample = self.version_space.sample_classifier(self.chain_length, self.sample_size,
                                                                           self._svm_classifier_sample)

    def update(self, points, labels):
        pass

    def predict(self, X):
        predictions = self._svm_classifier_sample.predict(X)
        return 2 * (np.sum(predictions, axis=0) > 0) - 1


class OptimalMargin(SamplingBase):
    def __init__(self, top=-1, chain_length=50, sample_size=8, cholesky=False, rounding=False, C=1000, kernel='rbf'):
        super().__init__(top=top, chain_length=chain_length, sample_size=sample_size, kernel=kernel, cholesky=cholesky, rounding=rounding)
        self.clf = SVC(C=C, kernel=kernel, decision_function_shape='ovr')
