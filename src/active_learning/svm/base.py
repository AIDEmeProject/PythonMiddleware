import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import rbf_kernel

from src.version_space import LinearVersionSpace
from src.convexbody.sampling import HitAndRunSampler
from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, top=-1, kind='linear', C=1000, kernel='linear', fit_intercept=True):
        super().__init__(top)

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, decision_function_shape='ovr')
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept)
        else:
            raise ValueError("Non supported kind. Only 'linear' and 'kernel' options available.")

        self.kind = kind


class SimpleMargin(SVMBase):
    """
        Picks the closest point to the decision boundary to feed the user. After Tong-and-Koller, this approximately
        cuts the version space in half at every iteration
    """
    def ranker(self, data):
        return np.abs(self.clf.decision_function(data))


class OptimalMargin(SVMBase):
    """
        Picks the point that cuts the version space approximately in half.
        Currently limited to linear kernel.
    """
    def __init__(self, threshold=-1, factorization_type='', chain_length=100, sample_size=-1, top=-1,
                 kind='linear', C=1000, kernel='linear', fit_intercept=True):
        super().__init__(kind=kind, kernel=kernel, C=C, fit_intercept=fit_intercept, top=top)
        self.threshold = threshold
        self.factorization_type = factorization_type
        self.chain_length = chain_length
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
        self.version_space = LinearVersionSpace(self.L.shape[1])

    def update(self, points, labels):
        super().update(self.L[points.index], labels)

    def ranker(self, data):
        samples = self.version_space.sample(self.chain_length, self.sample_size)
        bias, weight = samples[:, [0]], samples[:, 1:]
        prediction = np.sign(bias + np.dot(weight, self.L.T))
        return np.abs(np.sum(prediction, axis=-1))



