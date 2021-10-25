#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.

from __future__ import annotations

from typing import Union, Sequence, TYPE_CHECKING, Optional

from .polytope import *

if TYPE_CHECKING:
    from ... import PartitionedDataset
    from ...utils.types import Partition


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
    def __init__(self, partition: Optional[Partition] = None, mode: Union[str, Sequence[str]] = 'persist', tol: float = 1e-12):
        super().__init__(factorized=True)

        if not partition:
            partition = [slice(None)]

        if isinstance(mode, str):
            mode = [mode] * len(partition)

        if len(partition) != len(mode):
            raise ValueError("Lists have incompatible sizes: {0} and {1}".format(len(partition), len(mode)))

        self.partition = partition
        self.polytope_models = [self.get_polytope(m, tol) for m in mode]

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
