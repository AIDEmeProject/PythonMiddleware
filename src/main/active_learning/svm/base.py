import numpy as np
from sklearn.svm import SVC, LinearSVC

from .sampling import SamplingBase
from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, top=-1, kind='linear', C=1000, kernel='rbf', fit_intercept=True, class_weight=None):
        super().__init__(top)

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, decision_function_shape='ovr', class_weight=class_weight)
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept, class_weight=class_weight)
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


class OptimalMargin(SamplingBase, SVMBase):
    def __init__(self, top=-1, chain_length=50, sample_size=8, cholesky=False,
                 kind='linear', C=1000, kernel='rbf', fit_intercept=True, class_weight=None):
        SamplingBase.__init__(self, top=top, chain_length=chain_length, sample_size=sample_size, cholesky=cholesky)
        SVMBase.__init__(self, top=top, kind=kind, C=C, kernel=kernel, fit_intercept=fit_intercept, class_weight=class_weight)
