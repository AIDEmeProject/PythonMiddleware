import numpy as np

from ..active_learner import ActiveLearner


class SubspaceLearner(ActiveLearner):
    def __init__(self, partition, learners, label_function='AND', probability_function='min', ranking_function='SQUARE'):
        self.partition = partition
        self.learners = learners
        self.label_function = self.__get_label_connector(label_function)
        self.probability_function = self.__get_proba_connector(probability_function)
        self.ranking_function = self.__get_ranking_connector(ranking_function)

    @classmethod
    def __get_ranking_connector(cls, ranking_function):
        return cls.__get_function(ranking_function, {
            'SUM': lambda score: np.sum(score, axis=1),
            'SQUARE': lambda score: np.sum(np.square(score), axis=1),
        })

    @classmethod
    def __get_label_connector(cls, label_function):
        return cls.__get_function(label_function, {
            'AND': lambda ys: np.all(ys, axis=1).astype('float'),
            'OR': lambda ys: np.any(ys, axis=1).astype('float'),
        })

    @classmethod
    def __get_proba_connector(cls, probability_function):
        return cls.__get_function(probability_function, {
            'MIN': lambda ps: np.min(ps, axis=1),
            'MAX': lambda ps: np.max(ps, axis=1),
            'MEAN': lambda ps: np.mean(ps, axis=1),
        })

    @staticmethod
    def __get_function(function, name_to_function_map):
        if isinstance(function, str):
            return name_to_function_map.get(function.upper())

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
        return self.label_function(self._predict_all(X))

    def _predict_all(self, X):
        return np.array([learner.predict(X[:, idx]) for (idx, learner) in zip(self.partition, self.learners)]).T

    def predict_proba(self, X):
        """
        Predict probability of class being positive for each data point x in X.

        :param X: data matrix
        :return: positive class probability
        """
        return self.probability_function(self._predict_proba_all(X))

    def _predict_proba_all(self, X):
        return np.array([learner.predict_proba(X[:, idx]) for (idx, learner) in zip(self.partition, self.learners)]).T

    def rank(self, X):
        """
        Ranking function returning an "informativeness" score for each data point x in X. The lower the score, the most
        informative the data point is.

        :param X: data matrix
        :return: scores array
        """
        return self.ranking_function(self._rank_all(X))

    def _rank_all(self, X):
        return np.array([learner.rank(X[:, idx]) for (idx, learner) in zip(self.partition, self.learners)]).T
