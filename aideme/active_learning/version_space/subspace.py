#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from typing import Optional, Dict

import numpy as np

from .base import KernelVersionSpace
from .categorical import CategoricalActiveLearner, MultiSetActiveLearner
from ..active_learner import FactorizedActiveLearner
from ..margin import SimpleMargin


class SubspaceLearner(FactorizedActiveLearner):
    def __init__(self, base_learner, partition=None, label_function='AND', probability_function='MIN', ranking_function='SQUARE'):
        """
        :param base_learner: a Factory method for ActiveLearner objects.

        :param partition: default attribute partitioning into subspaces. If None, a single partition is assumed.

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
        self.base_learner = base_learner
        self.label_function = self.__get_label_connector(label_function)
        self.probability_function = self.__get_proba_connector(probability_function)
        self.ranking_function = self.__get_ranking_connector(ranking_function)
        self.__set_partition(partition)

    def set_factorization_structure(self, **factorization_info):
        """
        Updates the attribute partitioning.

        :param factorization_info: is expected to contain the key 'partition' with the new attribute partitioning. If
        not present, a single partition will be assumed.
        """
        partition = factorization_info.get('partition', None)
        self.__set_partition(partition)

    def __set_partition(self, partition):
        if not partition:
            partition = [slice(None)]

        self.partition = partition
        self.learners = [self.base_learner.clone() for _ in self.partition]

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
    def __init__(self, partition=None, mode='numerical', label_function='PROD', loss='GREEDY',
                 single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, rounding_cache: bool = True, rounding_options: Optional[Dict] = None,
                 add_intercept: bool = True, decompose: bool = False,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        """
        :param partition: default attribute partitioning into subspaces. If None, a single partition is assumed.

        :param mode: list of flags specifying the type of version space algorithm to use in each subspace. There are
        three possible cases:
                1) 'numerical': usual VS algorithm over numerical data
                2) 'categorical': special VS for the case where all attributes are categorical.
                3) 'multiset': special VS for the case where attributes come from a multi-set encoding.

        If a single string value is specified, the same mode will be assumed for all subspaces.

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

        base_learner = Cloneable(
            KernelVersionSpace,
            single_chain=single_chain, n_samples=n_samples, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding, rounding_cache=rounding_cache, rounding_options=rounding_options,
            add_intercept=add_intercept, decompose=decompose,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        label_function, probability_function = self.__get_proba_functions(label_function)

        super().__init__(base_learner=base_learner, label_function=label_function,
                         probability_function=probability_function, ranking_function=self.__get_loss_function(loss))

        self.set_factorization_structure(partition=partition, mode=mode)

    def set_factorization_structure(self, **factorization_info):
        """
        Updates the attribute partitioning and modes for each subspace.

        :param factorization_info: is expected contain two keys:

            - 'partition': the new factorization structure
            - 'mode': the list of modes to be used in each subspace. If it is a string, the same mode will be assumed in
            all subspaces.
        """
        super().set_factorization_structure(**factorization_info)

        size = len(self.partition)
        mode = factorization_info['mode']

        if isinstance(mode, str):
            mode = [mode] * size

        if len(mode) != size:
            raise ValueError("'mode' and 'partition' parameters have incompatible lengths.")

        self.learners = [self.__get_learner(m, learner) for m, learner in zip(mode, self.learners)]

    @staticmethod
    def __get_learner(mode, learner):
        mode = mode.upper()
        if mode in ('NUMERICAL', 'POSITIVE', 'NEGATIVE', 'PERSIST'):  # TODO: remove this coupling with DSM options
            return learner
        if mode == 'CATEGORICAL':
            return CategoricalActiveLearner()
        if mode == 'MULTISET':
            return MultiSetActiveLearner()

        raise ValueError("Unknown mode {}. Possible values are 'numerical', 'categorical', and 'multiset'.")

    @staticmethod
    def __get_proba_functions(label_function):
        if isinstance(label_function, str):
            label_function = label_function.upper()

        if label_function == 'AND':
            return 'AND', 'MIN'

        if label_function == 'OR':
            return 'OR', 'MAX'

        if label_function == 'PROD':
            return None, 'PROD'

        return label_function, None

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
    def __init__(self, partition=None, label_function='AND', C=1.0, kernel='rbf', gamma='auto'):
        """
        :param partition: default attribute partitioning into subspaces. If None, a single partition is assumed.

        :param label_function: Possible values are
                - 'AND': assume conjunctive connector, i.e. return 1 iff all partial labels are 1
                - 'OR': assume disjunction connector, i.e. return 1 iff any partial label is 1
                - Any callable computing the final labels from a matrix of partial labels (n_partitions x n_points)
        """
        base_learner = Cloneable(SimpleMargin, C=C, kernel=kernel, gamma=gamma)
        super().__init__(base_learner=base_learner, partition=partition,
                         label_function=label_function, probability_function=None, ranking_function='SUM')


class Cloneable:
    def __init__(self, klass, **params):
        self.klass = klass
        self.params = params

    def clone(self):
        return self.klass(**self.params)