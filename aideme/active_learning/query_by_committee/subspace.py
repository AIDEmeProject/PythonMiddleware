import numpy as np

from .base import KernelQueryByCommittee
from .categorical import CategoricalActiveLearner
from ..active_learner import ActiveLearner
from ..svm import SimpleMargin


class SubspaceLearner(ActiveLearner):
    def __init__(self, partition, learners, label_function='AND', probability_function='MIN', ranking_function='SQUARE'):
        """
        :param partition: attribute partitioning into subspaces

        :param learners: list of AL to run in each subspace

        :param label_function: Possible values are
                - None: make predictions by evaluating self.predict_proba(X) > 0.5
                - 'AND': assume conjunctive connector, i.e. return 1 iff all partial labels are 1
                - 'OR': assume disjunction connector, i.e. return 1 iff any partial label is 1
                - Any callable computing the final labels from a matrix of partial labels (n_partitions x n_points)

        :param probability_function: Possible values are
                - 'MIN': return min of all partial probability
                - 'MAX': return max of all partial probabilities
                - 'PROD': return the product of all partial probabilities
                - Any callable computing the final labels from a matrix of partial labels (n_partitions x n_points)

        :param ranking_function: Possible values are:
                - 'SUM': return the sum of partial ranks
                - 'SQUARE': return the squared sum of partial ranks
                - Any callable computing the final ranks from a matrix of partial ranks (n_partitions x n_points)
        """
        if len(partition) != len(learners):
            raise ValueError("Partition and learners must have the same size")

        self.partition = partition
        self.learners = learners
        self.label_function = self.__get_label_connector(label_function)
        self.probability_function = self.__get_proba_connector(probability_function)
        self.ranking_function = self.__get_ranking_connector(ranking_function)

    @classmethod
    def __get_label_connector(cls, connection):
        return cls._get_function(connection, {
            'AND': lambda ys: np.min(ys, axis=0),
            'OR': lambda ys: np.max(ys, axis=0),
        })

    @classmethod
    def __get_proba_connector(cls, probability_function):
        return cls._get_function(probability_function, {
            'MIN': lambda ps: ps.min(axis=0),
            'MAX': lambda ps: ps.max(axis=0),
            'PROD': lambda ps: ps.prod(axis=0),
        })

    @classmethod
    def __get_ranking_connector(cls, ranking_function):
        return cls._get_function(ranking_function, {
            'SUM': lambda score: score.sum(axis=0),
            'SQUARE': lambda score: np.square(score).sum(axis=0),
        })

    @staticmethod
    def _get_function(function, name_to_function_map):
        if isinstance(function, str):
            return name_to_function_map[function.upper()]

        if function is None or callable(function):
            return function

        raise ValueError("Expected callable or string, but received a " + type(function))

    def clear(self):
        for learner in self.learners:
            learner.clear()

    def fit(self, X, y):
        """
        Fit model over labeled data.

        :param X: data matrix
        :param y: labels array
        """
        # TODO: how to implement an equivalent version of the fit_data() method?
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
    def __init__(self, partition, label_function='AND', loss='GREEDY', categorical=None,
                 sampling='deterministic', n_samples=8, warmup=100, thin=10, sigma=100, rounding=True, add_intercept=True,
                 kernel='rbf', gamma=None, degree=3, coef0=0.):
        """
        :param categorical: indexes of partitions to treat as purely categorical ones. A special categorical AL can be
        trained in these cases, which usually improves the overall performance.

        :param label_function: Possible values are
                - 'AND': assume conjunctive connector, i.e. return 1 iff all partial labels are 1
                - 'OR': assume disjunction connector, i.e. return 1 iff any partial label is 1
                - 'PROD': compute probabilities via the 'PROD' connector
                - Any callable computing the final labels from a matrix of partial labels (n_partitions x n_points)

        :param loss: loss function used to select the next point to be labeled. Possible values are:
                - 'GREEDY': prod(1 - 2 p_k (1 - p_k))
                - 'SQUARED': sum_k (p_k - 0.5)^2
                - 'PRODUCT': | (prod_k p_k) - 0.5 |. This assumes assumes either 'AND' label function or 'PROD' probability function
                - Any callable computing the final ranks from a matrix of partial probabilities (n_partitions x n_points)
        """
        if categorical is None:
            categorical = []

        valid_idx = range(len(partition))
        if not set(categorical).issubset(valid_idx):
            raise ValueError("Invalid indexes found in 'categorical' parameter.")

        learners = [
            CategoricalActiveLearner() if i in categorical else
            KernelQueryByCommittee(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling, warmup=warmup,
                                   thin=thin, sigma=sigma, rounding=rounding,
                                   kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
            for i in valid_idx
        ]

        label_function, probability_function = self.__get_proba_functions(label_function)

        super().__init__(partition=partition, learners=learners, label_function=label_function,
                         probability_function=probability_function, ranking_function=self.__get_loss_function(loss))

    @staticmethod
    def __get_proba_functions(label_function):
        if isinstance(label_function, str):
            label_function = label_function.upper()

        if label_function == 'AND':
            return ('AND', 'MIN')

        if label_function == 'OR':
            return ('OR', 'MAX')

        if label_function == 'PROD':
            return (None, 'PROD')

        return (label_function, None)

    @classmethod
    def __get_loss_function(cls, loss):
        return super()._get_function(loss, {
            'GREEDY': lambda ps: np.prod(1 - 2 * ps * (1 - ps), axis=0),
            'SQUARE': lambda ps: np.square(ps - 0.5).sum(axis=0),
            'PRODUCT': lambda ps: np.abs(ps.prod(axis=0) - 0.5),
        })

    def rank(self, X):
        return self.ranking_function(self.predict_proba_all(X))  # ranking function depends on probabilities, not ranks


class SubspatialSimpleMargin(SubspaceLearner):
    def __init__(self, partition, label_function='AND', C=1.0, kernel='rbf', gamma='auto'):
        """
        :param label_function: Possible values are
                - 'AND': assume conjunctive connector, i.e. return 1 iff all partial labels are 1
                - 'OR': assume disjunction connector, i.e. return 1 iff any partial label is 1
                - Any callable computing the final labels from a matrix of partial labels (n_partitions x n_points)
        """
        learners = [SimpleMargin(C=C, kernel=kernel, gamma=gamma) for _ in partition]

        super().__init__(partition=partition, learners=learners,
                         label_function=label_function, probability_function=None, ranking_function='SUM')
