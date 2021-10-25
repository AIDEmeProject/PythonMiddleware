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

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .utils import assert_positive_integer

if TYPE_CHECKING:
    from .explore import LabeledSet
    from .utils import InitialSampler


__all__ = ['stratified_sampler', 'fixed_sampler', 'random_sampler']


def stratified_sampler(true_labels: LabeledSet, pos: int = 1, neg: int = 1, neg_in_all_subspaces: bool = False) -> InitialSampler:
    """
    Binary stratified sampling method. Randomly selects a given number of positive and negative points from an array
    of labels.

    :param pos: Number of positive points to sample. Must be non-negative.
    :param neg: Number of negative points to sample. Must be non-negative.
    """
    assert_positive_integer(pos, 'pos')
    assert_positive_integer(neg, 'neg')

    pos_mask = (true_labels.labels == 1)
    neg_mask = (true_labels.partial.max(axis=1) == 0) if neg_in_all_subspaces else ~pos_mask

    pos_idx, neg_idx = true_labels.index[pos_mask], true_labels.index[neg_mask]

    def sampler(data) -> Sequence:
        pos_samples = np.random.choice(pos_idx, size=pos, replace=False)
        neg_samples = np.random.choice(neg_idx, size=neg, replace=False)

        return list(pos_samples) + list(neg_samples)

    return sampler


def fixed_sampler(indexes: Sequence) -> InitialSampler:
    """
    Dummy sampler which returns a specified selection of indexes.
    """
    return lambda data: indexes


def random_sampler(sample_size: int) -> InitialSampler:
    """
    Samples a random batch of unlabeled points.
    """
    assert_positive_integer(sample_size, 'sample_size')
    return lambda data: data.unlabeled.sample(sample_size).index
