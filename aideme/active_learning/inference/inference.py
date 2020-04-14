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
from typing import Tuple

import numpy as np

from aideme.active_learning.dsm.polytope import CategoricalPolytope


# TODO: how to treat continuous variables?
# TODO: what to do in the case of contradictory labeling?
# TODO: is it possible to extend to the non-conjunctive case?
class Inference:
    def __init__(self, partition):
        self.partition = partition
        self._subspatial_inference = [CategoricalPolytope() for _ in self.partition]
        self._negative_point_cache = []

    def __repr__(self):
        return "neg_samples = {0}\n{1}".format(self._negative_point_cache, '\n'.join("subspace = {}, {}".format(p, s.__repr__()) for p, s in zip(self.partition, self._subspatial_inference)))

    @property
    def is_valid(self) -> bool:
        return all(s.is_valid for s in self._subspatial_inference)

    def clear(self) -> None:
        for s in self._subspatial_inference:
            s.clear()

        self._negative_point_cache = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_partial(X).min(axis=1)

    def predict_partial(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([subspace.predict(X[:, p]) for (p, subspace) in zip(self._subspatial_inference, self.partition)]).T

    def _predict_single(self, x: np.ndarray):
        return [subspace._predict_single(x[p]) for (p, subspace) in zip(self._subspatial_inference, self.partition)]

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.atleast_2d(X)

        for pt, lb in zip(X, y):
            self._update_single(pt, lb)

        if not self.is_valid:
            raise RuntimeError("Found contradicting labels in some subspace.")

    def _update_single(self, x: np.ndarray, y: float) -> None:
        # for positive points, we infer all partial labels as positive (conjunctive property)
        if y == 1:
            for p, subspace in zip(self.partition, self._subspatial_inference):
                subspace._update_single(x[p], 1)

            self.__attempt_to_remove_neg_samples()

        else:
            is_negative, has_negative_partial_labels = self.__can_infer_neg_point_label(x)

            if not is_negative:
                # if point's final label cannot be inferred from known labels, add it to neg point cache
                self._negative_point_cache.append(x)

            elif not has_negative_partial_labels:
                # when a new negative label is inferred, we can possibly remove some points from the cache
                self.__attempt_to_remove_neg_samples()

    def __attempt_to_remove_neg_samples(self) -> None:
        """
        Attempt to remove non-informative points from the negative point cache. A point is removed from cache whenever its
        final labels can successfully be inferred as negative.
        """
        to_remove, run_again = set(), False

        for i, x in enumerate(self._negative_point_cache):
            is_negative, has_negative_partial_labels = self.__can_infer_neg_point_label(x)

            if is_negative:
                # if the point's label can be successfully inferred, we no longer need to keep it
                to_remove.add(i)

                if not has_negative_partial_labels:
                    # in this case a new partial label was inferred, so we must run the removal process again
                    run_again = True

        if to_remove:
            self._negative_point_cache = [x for i, x in enumerate(self._negative_point_cache) if i not in to_remove]

        if run_again:
            self.__attempt_to_remove_neg_samples()

    def __can_infer_neg_point_label(self, x: np.ndarray) -> Tuple[bool, bool]:
        """
        For a give data point 'x', computes a boolean pair (is_negative, has_inferred). 'is_negative' tells whether 'x'
        label can be successfully inferred as negative. 'has_negative_partial_labels' tells whether any of 'x' partial
        labels is known to be negative.
        """
        idx_unknown, n_unknown = -1, 0

        for i, (p, subspace) in enumerate(zip(self.partition, self._subspatial_inference)):
            y_partial = subspace._predict_single(x[p])

            if y_partial == 0:
                return True, True

            elif y_partial == 0.5:
                idx_unknown = i
                n_unknown += 1

        if n_unknown == 1:
            # a single unknown partition remains, so its partial label must be negative
            self._subspatial_inference[idx_unknown]._update_single(x[self.partition[idx_unknown]], 0)
            return True, False

        if n_unknown == 0:
            raise RuntimeError("Conjunctive property violation: all partial labels are positive, but final label is negative")

        return False, False
