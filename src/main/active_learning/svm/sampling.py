import numpy as np
from src.main.experiments.utils import get_generator_average
from src.main.active_learning.base import ActiveLearner
from src.main.version_space import KernelVersionSpace


class SamplingBase(ActiveLearner):
    def __init__(self, top=-1, chain_length=50, sample_size=8, kernel='rbf', cholesky=False):
        super().__init__(top)

        self.cholesky = cholesky
        self.version_space = KernelVersionSpace(kernel, cholesky)

        self.sample_size = sample_size
        self.chain_length = chain_length
        self._samples = None

    def clear(self):
        super().clear()
        self._samples = None

    def update(self, points, labels):
        self.version_space.update(points, labels)

        # get initial point from previous samples
        if self._samples is None:
            initial_point = None
        else:
            initial_points = filter(self.version_space.is_inside, map(lambda x: np.hstack([x, [0]]), self._samples.get_params()))
            initial_point = get_generator_average(initial_points)

        #self._samples = self.version_space.sample(self.chain_length, self.sample_size, initial_point)
        self._samples = self.version_space.sample_classifiers(self.chain_length, self.sample_size, initial_point)

    def ranker(self, data):
        #bias, weight = self.get_bias_and_weight()
        #K = self.version_space.compute_kernel_against_data(data)
        #predictions = np.sign(bias + weight.dot(K.T))
        #predictions = (sample.predict(data) for sample in self._samples)
        predictions = self._samples.predict(data)
        return np.abs(np.sum(predictions, axis=0))

    def get_bias_and_weight(self):
        bias, weight = self._samples[:, 0].reshape(-1, 1), self._samples[:, 1:]

        if self.cholesky:
            weight = weight.dot(np.linalg.inv(self.version_space._L))
        return bias, weight


class MajorityVote(SamplingBase):
    def fit_classifier(self, X, y):
        pass

    def predict(self, X):
        bias, weight = self.get_bias_and_weight()

        K = self.version_space.compute_kernel_against_data(X)
        predictions = np.sign(bias + weight.dot(K.T))
        return 2*(np.sum(predictions, axis=0) > 0) - 1