import numpy as np
import sklearn.utils


class ActiveLearner:
    """
    Pool-based Active Learning base class. Active Learning object must be able to perform 2 main tasks:

        - Train a classification model over labeled data, capable of predicting class labels and, possibly, class probabilites.
        - Rank unlabeled points from "most informative" to "least informative"

    With also implement two helper methods: get_next, which returns which point the model considers the most informative;
    and run, which simulated the active learning main feedback look for a given number of iterations.
    """
    def fit(self, X, y):
        """
        Fit model over labeled data
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
        Ranking function returning an "informativeness" score for each data point x in X
        :param X: data matrix
        :return: scores array
        """
        raise NotImplementedError

    def get_next(self, X, labeled_index=None):
        """
        Get next point to label. We retrieve the "lowest score unlabeled point" in the dataset X.
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

    def run(self, X, y, n_iter, initial_sampler, metric=None):
        """
        Run Active Learning model over data, for a given number of iterations.
        :param X: data matrix
        :param y: labels array
        :param n_iter: number of iterations to run (after initial sampling)
        :param initial_sampler: InitialSampler object
        :param metric: accuracy metric to be computed every iteration
        :return: labeled points chosen by the algorithm
        """
        X, y = sklearn.utils.check_X_y(X, y)

        if len(np.unique(y)) > 2:
            raise ValueError("Found more than two distinct values in y; only binary classification is supported!")

        # fit model over initial sample
        labeled_indexes = initial_sampler(y)

        self.fit(X[labeled_indexes], y[labeled_indexes])

        # run n_iter iterations
        for i in range(n_iter):
            # get next point to label
            idx = self.get_next(X, labeled_indexes)
            labeled_indexes.append(idx)

            # fit model
            self.fit(X[labeled_indexes], y[labeled_indexes])

            # compute metric
            if metric:
                print('Iter', i, ': accuracy =', metric(y, self.predict(X)))

        return labeled_indexes
