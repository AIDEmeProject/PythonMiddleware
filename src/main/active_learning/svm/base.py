import numpy as np
from sklearn.svm import SVC, LinearSVC

from ..base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, kind='kernel', C=1000, kernel='rbf', fit_intercept=True, class_weight=None):
        super().__init__()

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


