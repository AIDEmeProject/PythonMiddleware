import numpy as np
from sklearn.utils import check_X_y


class LabeledSet(object):
    """
        In-memory dataset of all labeled points so far. Classification methods use the labeled set for training. 
    """
    def __init__(self):
        self.points = []
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def is_empty(self):
        return len(self.labels) == 0

    def append(self, points, labels):
        X, y = np.atleast_2d(points.data), np.atleast_1d(labels)
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)  # check consistent lengths, finite values

        if not self.is_empty() and X.shape[1] != len(self.points[-1]):
            raise ValueError(
                "Expeted {0}-dimensional points, "
                "obtained {1}-dimensional data".format(len(self.points[-1]), X.shape[1])
            )

        if not set(y) <= {-1, 1}:
            raise ValueError("All labels must be either 1 or -1")

        self.points.extend(X)
        self.labels.extend(y)

    def to_array(self):
        return np.array(self.points), np.array(self.labels)

