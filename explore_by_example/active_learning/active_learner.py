import numpy as np

class ActiveLearner:
    """
    Pool-based Active Learning base class. It performs two main tasks:

        - Trains a classification model over labeled data, predicting class labels and, possibly, class probabilities.
        - Ranks unlabeled points from "more informative" to "less informative"
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

    def next_points_to_label(self, idx, X):
        """
        Returns a list of data points to be labeled at the next iteration. By default, it returns a minimizer of the
        rank function at random.

        :param X: data matrix
        :return: row indexes of data points to be labeled
        """
        ranks = self.rank(X)
        min_row = np.arange(len(X))[ranks == ranks.min()]
        chosen_row = np.random.choice(min_row)
        return [idx[chosen_row]], X[[chosen_row]]
