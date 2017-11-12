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
        if self.kind == 'linear':
            self.version_space = SVMVersionSpace(data.shape[1])


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
    def __init__(self, C=1000, fit_intercept=True, chain_length=1000):
        super().__init__(kind='linear', C=C, fit_intercept=fit_intercept)
        self.sampler = HitAndRunSampler(chain_length)

    def sample_direction(self):
        initial_point = self.version_space.get_point()
        return self.sampler.sample_chain(self.version_space, initial_point)

    def ranker(self, data):
        samples = self.sample_direction()
        prediction = np.sign(data.dot(samples.T))
        #print(np.abs(np.sum(prediction, axis=-1)))
        return np.abs(np.sum(prediction, axis=-1))

