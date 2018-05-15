import numpy as np
import sklearn.utils


class ActiveLearner:
    """
    Pool-based Active Learning base class. It performs 2 main tasks:

        - Train a classification model over labeled data, capable of predicting class labels and, possibly, class probabilites.
        - Rank unlabeled points from "more informative" to "less informative"

    It also implements the helper methods "get_next", which returns which point the model considers the most informative
    from a given data pool (used for pool-based AL).
    """
    def fit(self, X, y):
        """
        Fit model over labeled data.

        :param X: data matrix
        :param y: labels array
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict classes for each data point x in X.

        :param X: data matrix
        :return: class labels
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Predict probability of class being positive for each data point x in X.

        :param X: data matrix
        :return: positive class probability
        """
        raise NotImplementedError

    def rank(self, X):
        """
        Ranking function returning an "informativeness" score for each data point x in X. The lower the score, the most
        informative the data point is.

        :param X: data matrix
        :return: scores array
        """
        raise NotImplementedError

    def get_next(self, X, labeled_index=None):
        """
        Get next point to label. We retrieve the "lowest ranked unlabeled point" in the dataset X.

        :param X: data matrix
        :param labeled_index: collection of previously labeled rows. These will not be considered when retrieving the
        next point.
        :return: row number of next point to be labeled
        """
        if labeled_index is None:
            labeled_index = []

        for row_number in np.argsort(self.rank(X)):
            if row_number not in labeled_index:
                return row_number

        raise RuntimeError("The entire dataset has already been labeled!")
