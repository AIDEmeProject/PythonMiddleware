"""
SVM-based Active Learning algorithms, from the article "Support Vector Machine Active Learning with Applications to Text
Classification", JMLR (2001), by Simon TONG & Daphne KOLLER.

Link: http://www.jmlr.org/papers/volume2/tong01a/tong01a.pdf
"""
import numpy as np
import sklearn
import sklearn.utils.validation

from .uncertainty import UncertaintySampler


class SimpleMargin(UncertaintySampler):
    """
    At every iteration, it trains an SVM model over labeled data, and picks the closest point to the decision boundary
    as most informative point.
    """
    def __init__(self, C=1.0, kernel='rbf'):
        UncertaintySampler.__init__(self, sklearn.svm.SVC(C=C, kernel=kernel))

    def rank(self, X):
        """
        Rank points based on their distance to the decision boundary.
        :param X: data matrix
        :return: distance list
        """
        return np.abs(self.clf.decision_function(X))


class RatioMargin(SimpleMargin):
    """
    For every unlabeled point, we train two SVM models: one positively biased and one negatively biased. The SVM's margin
    is an estimate for the Version Space remaining volume; thus, the point that most closely halves the current version
    space is the one whose two previously computed margins are the closest to each other.
    """
    def fit(self, X, y):
        self.clf.fit(X, y)

        # store training data
        self.X = X.copy()
        self.y = y.copy()

    def rank(self, X):
        # check model is fitted
        sklearn.utils.validation.check_is_fitted(self.clf, 'support_')

        scores = []
        classes = self.clf.classes_
        clf = sklearn.clone(self.clf)

        for x in X:
            data = np.vstack([self.X, x])

            # fit first model
            clf.fit(data, np.hstack([self.y, classes[0]]))
            margin0 = float(clf.dual_coef_.dot(clf.decision_function(clf.support_vectors_)))

            # fit second model
            clf.fit(data, np.hstack([self.y, classes[1]]))
            margin1 = float(clf.dual_coef_.dot(clf.decision_function(clf.support_vectors_)))

            # margin ratio
            if margin0 <= 0 or margin1 <= 0:
                scores.append(float('inf'))
            else:
                scores.append(-min(margin1/margin0, margin0/margin1))  # use negative sign since we want the LARGEST ratio

        return np.array(scores)
