from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, f1_score
from ..version_space.base import VersionSpace


class ActiveLearner:
    """ 
        Class responsible for two major points:
            - define which point to retrieve at every iteration
            - predicting the labels for other points of the data pool
    """
    def __init__(self, top=-1):
        self.clf = None
        self.version_space = VersionSpace()
        self.top = int(top)

    def predict(self, X):
        return self.clf.predict(X)

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def score(self, X, y_true):
        # classification scores
        y_pred = self.predict(X)
        scores = {
            'Precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
            'Recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F-Score': f1_score(y_true, y_pred, pos_label=1, average='binary')
        }

        return scores

    def clear(self):
        self.version_space.clear()

    def initialize(self, data):
        pass

    def update(self, points, labels):
        self.version_space.update(points, labels)

    def get_next(self, pool):
        return pool.get_minimizer_over_unlabeled_data(self.ranker, sample_size=self.top)

    def ranker(self, data):
        raise NotImplementedError
