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

from typing import TYPE_CHECKING

from aideme.utils import assert_positive_integer
from .active_learner import ActiveLearner

if TYPE_CHECKING:
    import numpy as np
    from .active_learner import IndexedDataset


class RandomSampler(ActiveLearner):
    """
    Randomly picks the next point to label. Usually used as baseline method for comparison.
    """
    def __init__(self, clf, batch_size: int = 1):
        """
        :param clf: Classifier object implementing two methods:
            - fit(X, y): fits the classifier over the labeled data X,y
            - predict(X): returns the class labels for a given set X

            Additionally, this object should implement predict_proba(X), but it is not mandatory.

        :param batch_size: number of random points to sample at every iteration. Default is 1.
        """
        assert_positive_integer(batch_size, 'batch_size')
        self._clf = clf
        self._batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    def _select_next(self, dataset: IndexedDataset) -> IndexedDataset:
        return dataset.sample(self._batch_size)
