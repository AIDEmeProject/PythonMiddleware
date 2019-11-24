import numpy as np


class ActiveLearner:
    """
    Pool-based Active Learning base class. It performs two main tasks:

        - Trains a classification model over labeled data, predicting class labels and, possibly, class probabilities.
        - Ranks unlabeled points from "more informative" to "less informative"
    """
    def clear(self):
        """
        Resets object internal state. Called at the beginning of each run.
        """
        pass

    def fit_data(self, data):
        """
        Trains active learning model over data

        :param data: PartitionedDataset object
        """
        X, y = data.training_set
        self.fit(X, y)

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

    def next_points_to_label(self, data, subsample=None):
        """
        Returns a list of data points to be labeled at the next iteration. By default, it returns a random minimizer of
        the rank function.

        :param data: a PartitionedDataset object
        :param subsample: size of unlabeled points sample. By default, no subsample is performed
        :return: row indexes of data points to be labeled
        """
        idx_sample, X_sample = data.unlabeled if subsample is None else data.sample_unlabeled(subsample)
        return self._select_next(idx_sample, X_sample)

    def _select_next(self, idx, X):
        ranks = self.rank(X)
        min_row = np.arange(len(X))[ranks == ranks.min()]
        chosen_row = np.random.choice(min_row)
        return [idx[chosen_row]], X[[chosen_row]]

    def set_factorization_structure(self, **factorization_info):
        """
        Tells the Active Learner to consider a new factorization structure, provided it can support such information

        :param partition: new attributes partitioning
        :param kwargs: any extra factorization information the AL may rely on
        """
        pass
