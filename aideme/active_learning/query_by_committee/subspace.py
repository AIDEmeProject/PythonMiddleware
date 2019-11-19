import numpy as np

from ..active_learner import ActiveLearner


class SubspaceLearner(ActiveLearner):
    def __init__(self, partition, learners, label_function='AND', probability_function='MIN', ranking_function='SQUARE'):
        if len(partition) != len(learners):
            raise ValueError("Partition and learners must have the same size")

        self.partition = partition
        self.learners = learners
        self.label_function = self.__get_label_connector(label_function)
        self.probability_function = self.__get_proba_connector(probability_function)
        self.ranking_function = self.__get_ranking_connector(ranking_function)

    @classmethod
    def __get_ranking_connector(cls, ranking_function):
        return cls.__get_function(ranking_function, {
            'SUM': lambda score: np.sum(score, axis=0),
            'SQUARE': lambda score: np.sum(np.square(score), axis=0),
        })

    @classmethod
    def __get_label_connector(cls, label_function):
        return cls.__get_function(label_function, {
            'AND': lambda ys: np.all(ys, axis=0).astype('float'),
            'OR': lambda ys: np.any(ys, axis=0).astype('float'),
        })

    @classmethod
    def __get_proba_connector(cls, probability_function):
        return cls.__get_function(probability_function, {
            'MIN': lambda ps: np.min(ps, axis=0),
            'MAX': lambda ps: np.max(ps, axis=0),
            'MEAN': lambda ps: np.mean(ps, axis=0),
        })

    @staticmethod
    def __get_function(function, name_to_function_map):
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
