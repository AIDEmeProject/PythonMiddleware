import numpy as np
from sklearn.svm import SVC, LinearSVC

from src.version_space import SVMVersionSpace
from src.convexbody.sampling import HitAndRunSampler
from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, kind='kernel', C=1000, kernel='rbf', degree=3, gamma='auto', max_iter=-1, fit_intercept=True):
        super().__init__()

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape='ovr', max_iter=max_iter)
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept)
        else:
            raise ValueError("Non supported kind. Only 'linear' and 'kernel' options available.")

        self.kind = kind

    def initialize(self, data):
        #if self.kind == 'linear':
        #    self.version_space = SVMVersionSpace(data.shape[1])
        #    self.version_space.clear()
        pass

class SimpleMargin(SVMBase):
    """
        Picks the closest point to the decision boundary to feed the user. After Tong-and-Koller, this approximately
        cuts the version space in half at every iteration
    """
    def get_next(self, pool):
        return pool.find_minimizer(lambda x: np.abs(self.clf.decision_function(x)))[1]


class OptimalMargin(SVMBase):
    """
        Picks the point that cuts the version space approximately in half.
        Currently limited to linear kernel
    """
    def __init__(self, C=1000, fit_intercept=False, chain_length=1000, known_labels=False):
        super().__init__(kind='linear', C=C, fit_intercept=fit_intercept)
        self.sampler = HitAndRunSampler(chain_length)
        self.known_labels = bool(known_labels)

    def get_next(self, pool):
        samples = self.sampler.sample_chain(self.version_space, self.version_space.get_point())
        known_labels, minimizer = pool.find_minimizer(
            ranker = lambda x: np.abs(np.sum(np.sign(x.dot(samples.T)), axis=-1)),
            threshold = lambda val: val == 0
        )
        if self.known_labels and known_labels is not None:
            pool.update(known_labels, self.predict(known_labels))
        return minimizer
