import numpy as np

from ..active_learner import ActiveLearner
from .base import KernelQueryByCommittee


class SubspaceLearner(ActiveLearner):
    def __init__(self, partition, learners, label_function='AND', probability_function='MIN', ranking_function='SQUARE'):
        if len(partition) != len(learners):
            raise ValueError("Partition and learners must have the same size")

        self.partition = partition
        self.learners = learners
        self.label_function = None if label_function is None else self.__get_label_connector(label_function)
        self.probability_function = self.__get_proba_connector(probability_function)
        self.ranking_function = self.__get_ranking_connector(ranking_function)

    @classmethod
    def __get_label_connector(cls, label_function):
        return cls._get_function(label_function, {
            'AND': lambda ys: np.min(ys, axis=0),
            'OR': lambda ys: np.max(ys, axis=0),
        })

    @classmethod
    def __get_proba_connector(cls, probability_function):
        return cls._get_function(probability_function, {
            'MIN': lambda ps: np.min(ps, axis=0),
            'MAX': lambda ps: np.max(ps, axis=0),
            'PROD': lambda ps: np.prod(ps, axis=0),
        })

    @classmethod
    def __get_ranking_connector(cls, ranking_function):
        return cls._get_function(ranking_function, {
            'SUM': lambda score: np.sum(score, axis=0),
            'SQUARE': lambda score: np.sum(np.square(score), axis=0),
        })

    @staticmethod
    def _get_function(function, name_to_function_map):
        if isinstance(function, str):
            return name_to_function_map[function.upper()]

        if callable(function):
            return function

        raise ValueError("Expected callable or string, but received a " + type(function))

    def fit(self, X, y):
        """
        Fit model over labeled data.

        :param X: data matrix
        :param y: labels array
        """
        for i, (idx, learner) in enumerate(zip(self.partition, self.learners)):
            learner.fit(X[:, idx], y[:, i])

    def predict(self, X):
        """
        Predict classes for each data point x in X.

        :param X: data matrix
        :return: class labels
        """
        if self.label_function is None:
            return (self.predict_proba(X) > 0.5).astype('float')

        return self.label_function(self.predict_all(X))

    def predict_proba(self, X):
        """
        Predict probability of class being positive for each data point x in X.

        :param X: data matrix
        :return: positive class probability
        """
        return self.probability_function(self.predict_proba_all(X))

    def rank(self, X):
        """
        Ranking function returning an "informativeness" score for each data point x in X. The lower the score, the most
        informative the data point is.

        :param X: data matrix
        :return: scores array
        """
        return self.ranking_function(self.rank_all(X))

    def predict_all(self, X):
        return self.__compute_over_all_subspaces(X, (l.predict for l in self.learners))

    def predict_proba_all(self, X):
        return self.__compute_over_all_subspaces(X, (l.predict_proba for l in self.learners))

    def rank_all(self, X):
        return self.__compute_over_all_subspaces(X, (l.rank for l in self.learners))

    def __compute_over_all_subspaces(self, X, funcs):
        values = np.empty( (len(self.partition), len(X)) )

        for i, (idx, func) in enumerate(zip(self.partition, funcs)):
            values[i] = func(X[:, idx])

        return values


class SubspatialVersionSpace(SubspaceLearner):
    def __init__(self, partition, label_function='AND', probability_function='MIN', loss='GREEDY',
                 sampling='deterministic', n_samples=8, warmup=100, thin=10, sigma=100, rounding=True, add_intercept=True,
                 kernel='linear', gamma=None, degree=3, coef0=0.):
        """
        :param partition: attribute partitioning into subspaces
        :param label_function: function used to piece together the predictions on each subspace into a final label. Possible values are:
                - None: simply return self.predict_proba(X) > 0.5
                - 'AND': assume conjunctive connector
                - 'OR': assume disjunctive connector
                - Any function f(ys) -> y_final, where ys is a matrix containing all partial labels

        :param loss: loss function used to select the next point to be labeled. Possible values are:
                - 'GREEDY': prod(1 - 2 p_k (1 - p_k))
                - 'SQUARED': sum_k (p_k - 0.5)^2
                - 'PRODUCT': | (prod_k p_k) - 0.5 | -> note it assumes a 'AND' labeling function
                - Any function f(ps) -> rank computing a rank from the estimated cut probabilities in each subspace
        """

        learners = [
            KernelQueryByCommittee(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling, warmup=warmup,
                                   thin=thin, sigma=sigma, rounding=rounding, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
            for _ in partition
        ]

        super().__init__(partition=partition, learners=learners,
                         label_function=label_function, probability_function=probability_function, ranking_function=self.__get_loss_function(loss))


    @classmethod
    def __get_loss_function(cls, loss):
        return super()._get_function(loss, {
            'GREEDY': lambda ps: np.prod(1 - 2 * ps * (1 - ps), axis=0),
            'SQUARED': lambda ps: np.square(ps - 0.5).sum(axis=0),
            'PRODUCT': lambda ps: np.abs(ps.prod(axis=0) - 0.5),
        })

    def rank(self, X):
        return self.ranking_function(self.predict_proba_all(X))  # ranking function depends on probabilities themselves
