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
    def __init__(self, chain_length=100, sample_size=-1, kind='linear', C=1000, kernel='linear', fit_intercept=True, top=None,
                 estimate_cut=False, threshold=-1, factorization_type=''):
        super().__init__(kind=kind, kernel=kernel, C=C, fit_intercept=fit_intercept, top=top, estimate_cut=estimate_cut)
        self.sampler = HitAndRunSampler(chain_length)
        self.threshold = threshold
        self.factorization_type = factorization_type
        self.sample_size = sample_size

    def get_factorization(self, data):
        if not self.factorization_type or self.kind == 'linear':
            return data.values
        if self.factorization_type == 'cholesky':
            K = rbf_kernel(data) + 1e-8 * np.eye(len(data))
            return np.linalg.cholesky(K)
        elif self.factorization_type == 'diagonalize':
            K = rbf_kernel(data) + 1e-8 * np.eye(len(data))
            lamb, P = np.linalg.eigh(K)
            L = P.dot(np.sqrt(np.diag(lamb)))
            threshold = -np.argmax(np.cumsum(lamb[::-1])/np.sum(lamb) > self.threshold)
            return L[:, threshold:]
        raise ValueError("Unknown factorization method")

    def initialize(self, data):
        self.L = self.get_factorization(data)
        print(self.L.shape)
        #self.L = np.hstack([np.ones((len(self.L), 1)), self.L])
        self.version_space = SVMVersionSpace(self.L.shape[1])
        #print(self.L.shape)

    def update(self, points, labels):
        index = points.index
        points, labels = np.atleast_2d(points.values), np.atleast_1d(labels).ravel()
        for ind, point, label in zip(index, points, labels):
            point = self.L[ind]
            self.version_space.update(point, label)

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        if self.sample_size > 0:
            return self.sampler.uniform(self.version_space, initial_point, self.sample_size)
        return self.sampler.sample_chain(self.version_space, initial_point)

    def ranker(self, data):
        samples = self.sample_direction()
        data = np.hstack([np.ones((len(self.L), 1)), self.L])
        prediction = np.sign(data.dot(samples.T))
        return np.abs(np.sum(prediction, axis=-1))
