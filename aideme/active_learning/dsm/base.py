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

import warnings
from typing import Optional, TYPE_CHECKING, Sequence, List, Tuple

import numpy as np

from .model import PolytopeModelBase
from ..active_learner import ActiveLearner

if TYPE_CHECKING:
    from ...explore.partitioned import PartitionedDataset, IndexedDataset


class DualSpaceModelBase(ActiveLearner):
    def __init__(self, polytope_model: PolytopeModelBase, active_learner: ActiveLearner, sample_unknown_proba: float = 0.5):
        self.active_learner = active_learner
        self.sample_unknown_proba = sample_unknown_proba
        self.polytope_model = polytope_model
        self._dsm_labeled_cache = TrainingSetCache()

    def clear(self) -> None:
        self.active_learner.clear()
        self.polytope_model.clear()
        self._dsm_labeled_cache.clear()

    def fit_data(self, data: PartitionedDataset) -> None:
        """
        Fits both active learner and polytope model.
        """
        self.__fit_active_learner(data)

        if not self.polytope_model.is_valid:
            return

        is_success = self.polytope_model.update_data(data)

        # if conflicting points were found, we must relabel the inferred partition and the DSM labeled points
        if not is_success:
            warnings.warn("Found conflicting point in polytope model. is_valid = {0}".format(self.polytope_model.is_valid))

            data.remove_inferred()
            self.__verify_dsm_labeled_points(data)

            # if polytope became invalid with the last update, skip relabeling
            if not self.polytope_model.is_valid:
                return

        if data.unknown_size > 0:
            unknown = data.unknown
            pred = self.polytope_model.predict(unknown.data)
            data.move_to_inferred(unknown.index[pred != 0.5])

    def __verify_dsm_labeled_points(self, data: PartitionedDataset) -> None:
        if self._dsm_labeled_cache.is_empty:
            return

        is_known = self.polytope_model.predict(self._dsm_labeled_cache.X) != 0.5

        if not np.all(is_known):
            self._dsm_labeled_cache.filter(is_known)
            self.__fit_active_learner(data)  # retrain AL since labeled changed

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts classes based on polytope model first; unknown labels are labeled via the active learner
        """
        if not self.polytope_model.is_valid:
            return self.active_learner.predict(X)

        predictions = self.polytope_model.predict(X)

        unknown_mask = (predictions == 0.5)

        if np.any(unknown_mask):
            predictions[unknown_mask] = self.active_learner.predict(X[unknown_mask])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities using the polytope model first; unknown labels are predicted via the active learner
        """
        if not self.polytope_model.is_valid:
            return self.active_learner.predict_proba(X)

        probas = self.polytope_model.predict(X)

        unknown_mask = (probas == 0.5)

        if np.any(unknown_mask):
            probas[unknown_mask] = self.active_learner.predict_proba(X[unknown_mask])

        return probas

    def rank(self, X: np.ndarray) -> np.ndarray:
        """
        Simply use AL to rank points
        """
        return self.active_learner.rank(X)

    def next_points_to_label(self, data: PartitionedDataset, subsample: Optional[int] = None) -> IndexedDataset:
        if not self.polytope_model.is_valid:
            return self.active_learner.next_points_to_label(data, subsample)

        while data.unknown_size > 0:
            sample = data.unknown.sample(subsample) if np.random.rand() < self.sample_unknown_proba else data.unlabeled.sample(subsample)
            selected = self.active_learner._select_next(sample)

            pred = self.polytope_model.predict(selected.data)
            is_known = (pred != 0.5)

            if np.any(is_known):
                self._dsm_labeled_cache.append(selected.data[is_known], pred[is_known])
                self.__fit_active_learner(data)

            if not np.all(is_known):
                return selected[~is_known]

        return self.active_learner.next_points_to_label(data, subsample)

    def __fit_active_learner(self, data: PartitionedDataset) -> None:
        X, y = data.training_set()

        if not self._dsm_labeled_cache.is_empty:
            X, y = self._dsm_labeled_cache.merge(X, y)

        self.active_learner.fit(X, y)


class TrainingSetCache:
    """
    Helper class for caching training data
    """
    def __init__(self):
        self._X: List[np.ndarray]
        self._y: List
        self._X, self._y = [], []

    @property
    def is_empty(self) -> bool:
        """
        :return: whether the cache is empty
        """
        return not self._X

    @property
    def X(self) -> np.ndarray:
        """
        :return: the training data as numpy array
        """
        return np.vstack(self._X)

    def clear(self) -> None:
        """
        Empties the cache
        """
        self._X, self._y = [], []

    def merge(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: a data array
        :param y: a labels array
        :return: a pair (X, y) containing the merge of input data and the cached data
        """
        return np.vstack([X, self._X]), np.hstack([y, self._y])

    def append(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Appends new data to the cache
        :param X: data matrix to append
        :param y: data labels to append
        """
        self._X.extend(X)
        self._y.extend(y)

    def filter(self, mask: Sequence[bool]) -> None:
        """
        Filters the cache elements with the given mask sequence
        :param mask: list of booleans indicating which positions to keep
        """
        self._X = [x for i, x in enumerate(self._X) if mask[i]]
        self._y = [y for i, y in enumerate(self._y) if mask[i]]
