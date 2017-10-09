import numpy as np
from sklearn.svm import SVC, LinearSVC
from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, kind='kernel', C=1000, kernel='rbf', degree=3, gamma='auto', max_iter=None, fit_intercept=True):
        super().__init__()

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape='ovr', max_iter=max_iter)
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept)
        else:
            raise ValueError("Non supported kind. Only 'linear' and 'kernel' options available.")

        self.kind = kind

    def initialize(self, data):
        if self.kind == 'linear':# and data.shape[1] == 2:
            from src.version_space.two_dimensional import Circle
            self.version_space = Circle()
            self.version_space.clear()


class SimpleMargin(SVMBase):
    def get_next(self, pool):
        return pool.find_minimizer(lambda x: np.abs(self.clf.decision_function(x)))
