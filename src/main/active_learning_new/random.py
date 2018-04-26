from numpy.random import permutation

from .active_learner import ActiveLearner


class RandomSampler(ActiveLearner):
    """
    Randomly picks the next point to label. Usually used as baseline method for comparison.
    """
    def __init__(self, clf):
        """
        :param clf: Classifier object implementing two methods:
            - fit(X, y): fits the classifier over the labeled data X,y
            - predict(X): returns the class labels for a given set X

            Additionally, this object can use implement predict_proba(X), but it is not mandatory.
        """
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def rank(self, X):
        return permutation(len(X))
