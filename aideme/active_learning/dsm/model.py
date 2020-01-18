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
from __future__ import annotations

from typing import Union, Sequence, TYPE_CHECKING

from .polytope import *

if TYPE_CHECKING:
    from ... import PartitionedDataset


class PolytopeModelBase(PolytopeBase):
    def __init__(self, factorized: bool):
        self._factorized = factorized

    @staticmethod
    def get_polytope(mode: str, tol: float) -> PolytopeBase:
        if mode == 'positive':
            return Polytope(tol)
        if mode == 'negative':
            return FlippedPolytope(tol)
        if mode == 'persist':
            return PersistentPolytope(tol)
        if mode == 'categorical':
            return CategoricalPolytope()
        if mode == 'multiset':
            return MultiSetPolytope()
        raise ValueError('Unknown mode {0}. Available values are: {1}'.format(mode, ['categorical', 'multiset', 'negative', 'persist', 'positive']))

    def update_data(self, data: PartitionedDataset) -> bool:
        X, y = data.last_training_set(get_partial=self._factorized)
        return self.update(X, y)


class PolytopeModel(PolytopeModelBase):
    def __init__(self, mode: str = 'persist', tol: float = 1e-12):
        super().__init__(factorized=False)
        self._pol = self.get_polytope(mode, tol)

    @property
    def is_valid(self) -> bool:
        return self._pol.is_valid

    def clear(self) -> None:
        self._pol.clear()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pol.predict(X)

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        return self._pol.update(X, y)


class FactorizedPolytopeModel(PolytopeModelBase):
    def __init__(self, partition: Sequence[Sequence[int]], modes: Union[str, Sequence[str]], tol: float = 1e-12):
        super().__init__(factorized=True)

        if isinstance(modes, str):
            modes = [modes] * len(partition)

        if len(partition) != len(modes):
            raise ValueError("Lists have incompatible sizes: {0} and {1}".format(len(partition), len(modes)))

        self.partition = partition
        self.polytope_models = [self.get_polytope(mode, tol) for mode in modes]

    @property
    def is_valid(self) -> bool:
        return any((pol.is_valid for pol in self.polytope_models))

    @property
    def is_all_valid(self) -> bool:
        return all((pol.is_valid for pol in self.polytope_models))

    def clear(self) -> None:
        for pol in self.polytope_models:
            pol.clear()

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not self.is_valid:
            raise RuntimeError("Cannot update invalid polytope.")

        return all(pol.update(X[:, idx], y[:, i]) for i, idx, pol in self.__valid_elements())

    def predict(self, X: np.ndarray) -> np.ndarray:
        val = 1.0 if self.is_all_valid else 0.5
        prediction = np.full(len(X), fill_value=val)

        for i, idx, pol in self.__valid_elements():
            np.minimum(prediction, pol.predict(X[:, idx]), out=prediction)

        return prediction

    def predict_partial(self, X: np.ndarray) -> np.ndarray:
        partial_labels = np.full((X.shape[0], len(self.partition)), fill_value=0.5)

        for i, idx, pol in self.__valid_elements():
            partial_labels[:, i] = pol.predict(X[:, idx])

        return partial_labels

    def __valid_elements(self):
        return ((i, idx, pol) for i, (idx, pol) in enumerate(zip(self.partition, self.polytope_models)) if pol.is_valid)
