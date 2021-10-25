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

import numpy as np

from .index import Index

if TYPE_CHECKING:
    from ..utils import Metrics


class LabeledSet:
    """
    This class manages a collection of user labels, including final labels, partial labels, and the corresponding data
    point indexes.
    """
    def __init__(self, labels, partial=None, index=None):
        """
        :param labels: the collection of user labels (0, 1 format)
        :param partial: the collection of user partial labels, as a matrix of 0, 1 values. Use None if there is no partial
        labels information from user.
        :param index: indexes corresponding to each label. If None, a range index will be assumed.
        """
        self.labels = np.ravel(labels)
        self.set_partial_labels(partial)
        self.index = self.__get_index(index)
        self.__index_to_row = Index(self.index)

    def set_partial_labels(self, partial) -> None:
        if partial is None:
            partial_labels = self.labels.reshape(-1, 1)

        else:
            partial_labels = np.asarray(partial)

            if partial_labels.ndim != 2:
                raise ValueError("Expected two-dimensional array of partial labels, but ndim = {}".format(partial_labels.ndim))

            if partial_labels.shape[0] != len(self):
                raise ValueError("Wrong size of partial_labels: expected {}, but got {}".format(partial_labels.shape[0], len(self)))

        self.partial = partial_labels

    def __get_index(self, index) -> np.ndarray:
        if index is None:
            return np.arange(len(self))

        idx = np.ravel(index)

        if len(idx) != len(self):
            raise ValueError("Wrong size of indexes: expected {}, but got {}".format(len(idx), len(self)))

        return idx

    def __len__(self):
        return len(self.labels)

    @property
    def num_partitions(self):
        return self.partial.shape[1]

    def get_index(self, idx):
        rows = self.__index_to_row.get_rows(idx)
        return LabeledSet(self.labels[rows], self.partial[rows], self.index[rows])

    def concat(self, labeled_set: LabeledSet) -> LabeledSet:
        if len(self) == 0:
            return labeled_set

        if len(labeled_set) == 0:
            return self

        labels = np.hstack([self.labels, labeled_set.labels])
        partial = np.vstack([self.partial, labeled_set.partial])
        index = np.hstack([self.index, labeled_set.index])
        return LabeledSet(labels, partial, index)

    def asdict(self, noisy: bool = False) -> Metrics:
        """
        :param noisy: whether to add 'noisy' prefix to labels
        :return: a dict containing all index and labels information
        """
        prefix = 'noisy_' if noisy else ''

        metrics = {
            prefix + 'labels': self.labels.tolist(),
            prefix + 'partial_labels': self.partial.tolist(),
        }

        if not noisy:
            metrics['labeled_indexes'] = self.index.tolist()

        return metrics

    def has_positive_and_negative_labels(self):
        return len(self.labels) > 0 and 0 < self.labels.sum() < len(self.labels)
