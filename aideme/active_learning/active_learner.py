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

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from aideme.utils import metric_logger

if TYPE_CHECKING:
    from aideme.explore import PartitionedDataset
    from aideme.explore.partitioned import IndexedDataset


class ActiveLearner:
    """
    Pool-based Active Learning base class. It performs two main tasks:

        - Trains a classification model over labeled data, predicting class labels and, possibly, class probabilities.
        - Ranks unlabeled points from "more informative" to "less informative"
    """
    def clear(self) -> None:
        """
        Resets object internal state. Called at the beginning of each run.
        """
        pass

    def fit_data(self, data: PartitionedDataset) -> None:
        """
        Trains active learning model over data

        :param data: PartitionedDataset object
        """
        X, y = data.training_set()
        self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model over labeled data.

        :param X: data matrix
        :param y: labels array
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for each data point x in X.

        :param X: data matrix
        :return: class labels
        """
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class being positive for each data point x in X.

        :param X: data matrix
        :return: positive class probability
        """
        raise NotImplementedError

    def rank(self, X: np.ndarray) -> np.ndarray:
        """
        Ranking function returning an "informativeness" score for each data point x in X. The lower the score, the most
        informative the data point is.

        :param X: data matrix
        :return: scores array
        """
        raise NotImplementedError

    def next_points_to_label(self, data: PartitionedDataset, subsample: Optional[int] = None) -> IndexedDataset:
        """
        Returns a list of data points to be labeled at the next iteration. By default, it returns a random minimizer of
        the rank function.

        :param data: a PartitionedDataset object
        :param subsample: size of unlabeled points sample. By default, no subsample is performed
        :return: row indexes of data points to be labeled
        """
        return self._select_next(data.unlabeled.sample(subsample))

    def _select_next(self, dataset: IndexedDataset) -> IndexedDataset:
        ranks = self.rank(dataset.data)
        min_rank = ranks.min()

        metric_logger.log_metric('min_rank', min_rank)

        min_row = np.where(ranks == min_rank)[0]
        return dataset[np.random.choice(min_row)]


class FactorizedActiveLearner(ActiveLearner):
    def fit_data(self, data: PartitionedDataset) -> None:
        X, y = data.training_set(get_partial=True)
        self.fit(X, y)

    def set_factorization_structure(self, **factorization_info: Any) -> None:
        """
        Tells the Active Learner to consider a new factorization structure, provided it can support such information

        :param partition: new attributes partitioning
        :param factorization_info: any extra factorization information the AL may rely on
        """
        pass
