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
from typing import Set, List, Tuple

import numpy as np

from .convex import PositiveRegion, NegativeCone, ConvexError


class PolytopeBase:
    @property
    def is_valid(self) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        raise NotImplementedError


class Polytope(PolytopeBase):
    def __init__(self, tol: float = 1e-12):
        self.__is_valid = True
        self.positive_region = PositiveRegion(tol)
        self.negative_region: List[NegativeCone] = []

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    def clear(self) -> None:
        self.positive_region.clear()
        self.negative_region = []
        self.__is_valid = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predicts the label of each data point. Returns 1 if point is in positive region, 0 if in negative region,
            and 0.5 otherwise

            :param X: data matrix to predict labels
        """
        X = np.atleast_2d(X)

        probas = np.full(len(X), fill_value=0.5)

        if self.__is_valid:
            if self.positive_region.is_built:
                probas[self.positive_region.is_inside(X)] = 1.0

            for cone in self.negative_region:
                if cone.is_built:
                    probas[cone.is_inside(X)] = 0.0

        return probas

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Increments the polytope model with new labeled data

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.__is_valid:
            raise RuntimeError("Cannot update invalid polytope.")

        X, y = np.atleast_2d(X), np.asarray(y)

        try:
            self.__update_positive(X[y == 1])
            self.__update_negative(X[y != 1])

        except ConvexError:
            self.__is_valid = False

        finally:
            return self.__is_valid

    def __update_positive(self, X: np.ndarray) -> None:
        if len(X) > 0:
            self.positive_region.update(X)

            for nr in self.negative_region:
                nr.add_points_to_hull(X)

    def __update_negative(self, X: np.ndarray) -> None:
        for x in X:
            cone = NegativeCone(x, self.positive_region._tol)
            cone.add_points_to_hull(self.positive_region.vertices)
            self.negative_region.append(cone)


class FlippedPolytope(PolytopeBase):
    """
    Polytope model built assuming the negative region is convex.
    """
    def __init__(self, tol: float = 1e-12):
        """
        :param tol: polytope model tolerance
        """
        self._pol = Polytope(tol)

    @property
    def is_valid(self) -> bool:
        return self._pol.is_valid

    def clear(self) -> None:
        self._pol.clear()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: data matrix to predict labels
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        return 1 - self._pol.predict(X)

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        return self._pol.update(X, 1 - y)


class PersistentPolytope(PolytopeBase):
    """
    Polytope model built assuming the negative region is convex.
    """
    def __init__(self, tol: float = 1e-12):
        """
        :param tol: polytope model tolerance
        """
        self._pol = Polytope(tol)
        self._flipped = FlippedPolytope(tol)

    @property
    def is_valid(self) -> bool:
        return self._pol.is_valid or self._flipped.is_valid

    def clear(self) -> None:
        self._pol.clear()
        self._flipped.clear()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: data matrix to predict labels
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        X = np.atleast_2d(X)

        if self._pol.is_valid == self._flipped.is_valid:
            return np.full(len(X), fill_value=0.5)

        if self._pol.is_valid:
            return self._pol.predict(X)

        return self._flipped.predict(X)

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        if self._pol.is_valid and self._flipped.is_valid:
            return self._pol.update(X, y) and self._flipped.update(X, y)

        if self._pol.is_valid:
            return self._pol.update(X, y)

        return self._flipped.update(X, y)


class CategoricalPolytope(PolytopeBase):
    """
    Special polytope for the case where all attributes are assumed to be categorical. It simply memorizes the positive and
    negative values seen so far.
    """
    def __init__(self):
        self._pos_classes: Set[Tuple] = set()
        self._neg_classes: Set[Tuple] = set()
        self.__is_valid = True

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    def clear(self) -> None:
        self._pos_classes = set()
        self._neg_classes = set()
        self.__is_valid = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: data array to predict labels.
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        if not self.__is_valid:
            return np.full(len(X), fill_value=0.5)

        return np.fromiter((self.__predict_single(x) for x in X), np.float)

    def __predict_single(self, x: np.ndarray) -> float:
        x = tuple(x)

        if x in self._pos_classes:
            return 1.0

        if x in self._neg_classes:
            return 0.0

        return 0.5

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        for pt, lb in zip(X, y):
            pt = tuple(pt)  # convert to tuple because numpy arrays are not hashable

            if lb == 1:
                self._pos_classes.add(pt)
            else:
                self._neg_classes.add(pt)

        self.__is_valid = len(self._pos_classes & self._neg_classes) == 0

        return self.__is_valid


class MultiSetPolytope(PolytopeBase):
    def __init__(self):
        self._pos_indexes: Set[int] = set()
        self._neg_indexes: Set[int] = set()
        self._neg_point_cache: List[Set[int]] = []
        self.__is_valid = True

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    def clear(self) -> None:
        self._pos_indexes = set()
        self._neg_indexes = set()
        self._neg_point_cache = []
        self.__is_valid = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: data array to predict labels.
        :return: predicted labels. 1 for positive, 0 for negative, 0.5 for unknown
        """
        pred = np.full(len(X), fill_value=0.5)

        if not self.__is_valid:
            return pred

        if self._pos_indexes:
            pos = list(self._pos_indexes)
            pred[X[:, pos].sum(axis=1) == X.sum(axis=1)] = 1.0

        if self._neg_indexes:
            neg = list(self._neg_indexes)
            pred[X[:, neg].any(axis=1)] = 0.0

        return pred

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Increments the polytopes with new labeled data.

        :param X: data matrix. We expect it to be a 0-1 encoding of the multi-set.
        :param y: labels array. Expects 1 for positive points, and 0 for negative points
        :return: whether update was successful or not
        """
        if not self.is_valid:
            raise RuntimeError("Attempting to update invalid polytope.")

        for pt, lb in zip(X, y):
            if not self.__update_single(pt, lb):
                self.__is_valid = False
                return False

        return True

    def __update_single(self, pt: np.ndarray, lb: int) -> bool:
        """
        Increments polytope with a single point.
        :param pt: data point
        :param lb: data point label
        :return: whether update was successful or not
        """
        idx = {i for i, ind in enumerate(pt) if ind > 0}

        if lb == 1:
            return self.__update_positive_set(idx)

        idx -= self._pos_indexes
        if len(idx) == 1:
            return self.__update_negative_set(idx)

        self._neg_point_cache.append(idx)
        return True

    def __update_positive_set(self, idx: Set[int]) -> bool:
        """
        Adds new indexes to the positive set. Also, removes the new positive indexes from each negative point in the cache.

        :param idx: set of new positive indexes
        """
        if not idx:
            return True

        if len(idx & self._neg_indexes) > 0:
            return False

        self._pos_indexes.update(idx)

        new_neg = set()

        for cache in self._neg_point_cache:
            cache.difference_update(idx)

            if len(cache) == 0:
                return False

            elif len(cache) == 1:
                new_neg.update(cache)

        return self.__update_negative_set(new_neg)

    def __update_negative_set(self, idx: Set[int]) -> bool:
        """
        Adds new indexes to the negative set. Also, if a cached negative points contains any of the new negative indexes,
        it will also be removed from cache since no inference can be done on its values.

        :param idx: set of new negative indexes
        """
        if not idx:
            return True

        if len(idx & self._pos_indexes) > 0:
            return False

        self._neg_indexes.update(idx)
        self._neg_point_cache = [neg_idx for neg_idx in self._neg_point_cache if len(idx & neg_idx) == 0]

        return True
