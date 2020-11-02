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
from math import ceil
from typing import Optional

import numpy as np

from aideme.utils import assert_positive_integer
from .linear import FactorizedLinearLearner


class FactorizationSelector:
    def __init__(self, max_partitions: Optional[int] = None, max_dim: Optional[int] = None):
        assert_positive_integer(max_partitions, 'max_partitions', allow_none=True)
        assert_positive_integer(max_dim, 'max_dim', allow_none=True)

        self._max_dim = max_dim
        self._max_partitions = max_partitions

    def __call__(self, X: np.ndarray, y: np.ndarray, learner: FactorizedLinearLearner):
        N = X.shape[1]
        max_dim, max_partitions = self.__get_params(N)

        if N == 1 or max_partitions == 1:
            return [list(range(N))]

        if max_dim == 1:
            return [[i] for i in range(N)]

        return self._select_best_partition(X, y, learner, max_dim, max_partitions)

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, learner: FactorizedLinearLearner, max_dim: int, max_partitions: int):
        raise NotImplementedError

    @staticmethod
    def _is_valid_partition(partition, max_dim: int) -> bool:
        return max(map(len, partition)) <= max_dim

    def __get_params(self, N: int):
        max_dim = N if self._max_dim is None else self.__get_min(N, self._max_dim)
        max_partitions = N if self._max_partitions is None else self.__get_min(N, self._max_partitions)

        if max_partitions * max_dim < N:
            raise ValueError(
                "There are no partitions satisfying N = {}, K_max = {} and D_max = {}".format(N, max_partitions, max_dim)
            )

        return max_dim, max_partitions

    @staticmethod
    def __get_min(dim: int, max_dim: Optional[int]) -> int:
        return max_dim if dim is None else min(dim, max_dim)


class BruteForceSelector(FactorizationSelector):

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, learner: FactorizedLinearLearner, max_dim: int, max_partitions: int):
        N = X.shape[1]
        min_partitions = ceil(N / max_dim)
        opt_partition, opt_loss = None, np.inf

        for k in range(min_partitions, max_partitions + 1):
            for partition in self.__generate_all_partitions(N, k):

                if self._is_valid_partition(partition, max_dim):
                    loss = learner.compute_factorization_loss(X, y, partition)
                    if loss < opt_loss:
                        opt_partition, opt_loss = partition, loss

        return opt_partition

    @classmethod
    def __generate_all_partitions(cls, N, K):
        # N >= 1, 1 <= D <= N, N / D <= K <= N
        l = list(range(N))
        for c in cls.__partitions(l, K):
            if all(x for x in c):
                yield c

    @classmethod
    def __partitions(cls, l, K):
        if l:
            prev = None
            for t in cls.__partitions(l[1:], K):
                tup = sorted(t)
                if tup != prev:
                    prev = tup
                    for i in range(K):
                        yield tup[:i] + [[l[0]] + tup[i], ] + tup[i + 1:]
        else:
            yield [[] for _ in range(K)]


class GreedySelector(FactorizationSelector):
    def __init__(self, max_partitions: Optional[int] = None, max_dim: Optional[int] = None, warm_start: bool = True):
        super().__init__(max_partitions, max_dim)
        self.warm_start = warm_start

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, learner: FactorizedLinearLearner, max_dim: int, max_partitions: int):
        N = X.shape[1]
        opt_clf, opt_loss = learner.fit_and_loss(X, y, [[i] for i in range(N)])

        for k in range(N):
            level_clf, level_loss = opt_clf, opt_loss

            for i in range(N - k):
                for j in range(i + 1, N - k):
                    merged_clf = opt_clf.merge_partitions(i, j, self.warm_start)

                    if self._is_valid_partition(merged_clf.partition, max_dim):
                        clf, loss = learner.fit_and_loss(X, y, merged_clf.partition, merged_clf.weights)
                        if loss < level_loss:
                            level_clf, level_loss = clf, loss

            if level_loss < opt_loss:
                opt_clf, opt_loss = level_clf, level_loss
            else:
                break

        return opt_clf.partition
