import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import rbf_kernel

from src.version_space import SVMVersionSpace
from src.convexbody.sampling import HitAndRunSampler
from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, kind='linear', C=1000, kernel='linear', fit_intercept=True, top=None, estimate_cut=False):
        super().__init__(estimate_cut)

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, decision_function_shape='ovr')
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept)
        else:
            raise ValueError("Non supported kind. Only 'linear' and 'kernel' options available.")

        self.top = top
        self.kind = kind

    def initialize(self, data):
        self.version_space = SVMVersionSpace(data.shape[1])

    def get_sample(self, data):
        if not self.top:
            return data
        idx = np.random.choice(np.arange(len(data)), size=self.top, replace=False)
        return data[idx]


class SimpleMargin(SVMBase):
    """
        Picks the closest point to the decision boundary to feed the user. After Tong-and-Koller, this approximately
        cuts the version space in half at every iteration
    """
    def ranker(self, data):
        data = self.get_sample(data)
        return np.abs(self.clf.decision_function(data))


class OptimalMargin(SVMBase):
    """
        Picks the point that cuts the version space approximately in half.
        Currently limited to linear kernel.
    """
    def __init__(self, chain_length=100, kind='linear', C=1000, kernel='linear', fit_intercept=True, top=None, estimate_cut=False):
        super().__init__(kind=kind, kernel=kernel, C=C, fit_intercept=fit_intercept, top=top, estimate_cut=estimate_cut)
        self.sampler = HitAndRunSampler(chain_length)

    def initialize(self, data):
        n = data.shape[1]
        if self.kind == 'kernel':
            K = rbf_kernel(data)
            self.L = np.linalg.cholesky(K + 1e-5 * np.eye(len(K)))
            n = len(self.L)

        self.version_space = SVMVersionSpace(n)

    def update(self, points, labels):
        index = points.index
        points, labels = np.atleast_2d(points.values), np.atleast_1d(labels).ravel()
        for ind, point, label in zip(index, points, labels):
            if self.kind == 'kernel':
                point = self.L[ind]
            self.version_space.update(point, label)

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        return self.sampler.sample_chain(self.version_space, initial_point)

    def get_sample(self, data):
        if self.kind == 'kernel':
            samples = super().get_sample(self.L)
        else:
            samples = super().get_sample(data)
        return np.hstack([np.ones((len(samples), 1)), samples])

    def ranker(self, data):
        data = self.get_sample(data)
        samples = self.sample_direction()
        prediction = np.sign(data.dot(samples.T))
        return np.abs(np.sum(prediction, axis=-1))


class IncrementalOptimalMargin(SVMBase):
    """
        Picks the point that cuts the version space approximately in half.
        Currently limited to linear kernel.
    """
    def __init__(self, chain_length=100, C=1000, kernel='linear', top=None, estimate_cut=False):
        super().__init__(kind='kernel', kernel=kernel, C=C, top=top, estimate_cut=estimate_cut)
        self.sampler = HitAndRunSampler(chain_length)

    def initialize(self, data):
        K = rbf_kernel(data)
        self.L = np.linalg.cholesky(K + 1e-5 * np.eye(len(K)))
        self.version_space = SVMVersionSpace(len(self.L))

    def update(self, points, labels):
        index = points.index
        points, labels = np.atleast_2d(points.values), np.atleast_1d(labels).ravel()
        for ind, point, label in zip(index, points, labels):
            point = self.L[ind]
            self.version_space.update(point, label)

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        return self.sampler.sample_chain(self.version_space, initial_point)

    def get_sample(self, data):
        samples = super().get_sample(self.L)
        return np.hstack([np.ones((len(samples), 1)), samples])

    def ranker(self, data):
        data = self.get_sample(data)
        samples = self.sample_direction()
        prediction = np.sign(data.dot(samples.T))
        return np.abs(np.sum(prediction, axis=-1))
