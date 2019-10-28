from sklearn.utils import check_random_state

from .active_learner import ActiveLearner


class RandomSampler(ActiveLearner):
    """
    Randomly picks the next point to label. Usually used as baseline method for comparison.
    """
    def __init__(self, clf, random_state=None):
        """
        :param clf: Classifier object implementing two methods:
            - fit(X, y): fits the classifier over the labeled data X,y
            - predict(X): returns the class labels for a given set X

            Additionally, this object can use implement predict_proba(X), but it is not mandatory.
        """
        self._clf = clf
        self.__random_state = check_random_state(random_state)

    def fit(self, X, y):
        self._clf.fit(X, y)

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def rank(self, X):
        return self.__random_state.permutation(len(X))
