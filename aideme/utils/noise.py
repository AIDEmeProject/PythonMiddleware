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
import numpy as np
from sklearn import svm

from aideme.explore.labeledset import LabeledSet
from .types import NoiseInjector

"""
This module possesses a few helper functions for simulating user noisy labeling. In general, a "noise injector" method
can be any function with the following signature:

        noise_injector(labeled_set: LabeledSet) -> LabeledSet

Which receives the original noise-free LabeledSet object and returns its noisy version.
"""


def gaussian_noise(positive, negative, sigma) -> NoiseInjector:
    # TODO: how to implement this?
    border = svm.SVC(C=10**7, kernel='rbf')
    border.fit(positive, negative)
        
    pass


def random_noise_injector(noise: float) -> NoiseInjector:
    """
    Adds random noise to all labels, i.e each labels is flipped with the same probability.
    :param noise: probability of flipping labels
    :return: a noise injector
    """
    if noise < 0 or noise > 0.5:
        raise ValueError("Noise must be between 0 and 0.5, but got {}".format(noise))

    def injector(labeled_set: LabeledSet) -> LabeledSet:
        noisy_labels = __flip(labeled_set.labels, noise)
        noisy_partial_labels = __flip(labeled_set.partial, noise) if labeled_set.partial.shape[1] > 1 else None
        return LabeledSet(noisy_labels, noisy_partial_labels, labeled_set.index)

    return injector


def __flip(labels, noise):
    flip_mask = np.random.rand(*labels.shape) < noise
    return np.where(flip_mask, 1 - labels, labels)
