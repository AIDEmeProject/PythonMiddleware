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
from .linear import FactorizedLinearClassifier


class FactorizationSelector:
    def __init__(self, max_partitions: Optional[int] = None, max_dim: Optional[int] = None):
        assert_positive_integer(max_partitions, 'max_partitions', allow_none=True)
        assert_positive_integer(max_dim, 'max_dim', allow_none=True)

        self._max_dim = max_dim
        self._max_partitions = max_partitions

    def __call__(self, X: np.ndarray, y: np.ndarray, clf: FactorizedLinearClassifier):
        N = X.shape[1]
        max_dim, max_partitions = self.__get_params(N)

        if N == 1 or max_partitions == 1:
            return [list(range(N))]

        if max_dim == 1:
            return [[i] for i in range(N)]

        return self._select_best_partition(X, y, clf, max_dim, max_partitions)

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, clf: FactorizedLinearClassifier, max_dim: int, max_partitions: int):
        raise NotImplementedError

    @staticmethod
    def _is_valid_partition(partition, max_dim: int) -> bool:
        return max(map(len, partition)) <= max_dim

    @staticmethod
    def _compute_loss(X: np.ndarray, y: np.ndarray, clf: FactorizedLinearClassifier, part) -> float:
        clf.set_partition(part)
        res = clf.fit(X, y)
        return res.fun

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

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, clf: FactorizedLinearClassifier, max_dim: int, max_partitions: int):
        N = X.shape[1]
        min_partitions = ceil(N / max_dim)
        opt_partition, opt_loss = None, np.inf

        for k in range(min_partitions, max_partitions + 1):
            for partition in self.__generate_all_partitions(N, k):

                if self._is_valid_partition(partition, max_dim):

                    loss = self._compute_loss(X, y, clf, partition)
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

    def _select_best_partition(self, X: np.ndarray, y: np.ndarray, clf: FactorizedLinearClassifier, max_dim: int, max_partitions: int):
        N = X.shape[1]
        opt_partition = [[i] for i in range(N)]
        opt_loss = self._compute_loss(X, y, clf, opt_partition)

        for k in range(N):
            level_loss, level_part = opt_loss, opt_partition

            for i in range(N - k):
                for j in range(i + 1, N - k):
                    part = self.__merge_partitions(opt_partition, i, j)

                    if self._is_valid_partition(part, max_dim):
                        loss = self._compute_loss(X, y, clf, part)
                        if loss < level_loss:
                            level_loss, level_part = loss, part

            if level_loss < opt_loss:
                opt_partition, opt_loss = level_part, level_loss
            else:
                break

        return opt_partition

    @staticmethod
    def __merge_partitions(part, i, j):
        if i == j:
            return part

        i, j = min(i, j), max(i, j)
        return part[:i] + part[i+1:j] + part[j+1:] + [sorted(set(part[i] + part[j]))]
