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

"""
This module possesses a few helper functions for simulating user noisy labeling. In general, a "noise injector" method
can be any function with the following signature:

        noise_injector(labeled_set: LabeledSet) -> LabeledSet

Which receives a noise-free LabeledSet object and returns its noisy version.
"""

import numpy as np

from aideme.explore.labeledset import LabeledSet
from .types import NoiseInjector
from .validation import assert_non_negative_integer


def gaussian_noise() -> NoiseInjector:
    # TODO: how to implement this?
    pass


def random_noise_injector(noise: float, skip_initial: int = 0) -> NoiseInjector:
    """
    Adds random noise to all labels, i.e each labels is flipped with the same probability.
    :param noise: probability of flipping labels
    :param skip_initial: number of initial iterations to skip the noise injection during exploration phase
    :return: a noise injector
    """
    if noise < 0 or noise > 1:
        raise ValueError("Noise must be between 0 and 1, but got {}".format(noise))

    def injector(iteration: int, labeled_set: LabeledSet) -> LabeledSet:
        noisy_labels = __flip(labeled_set.labels, noise)

        noisy_partial_labels = None
        if labeled_set.num_partitions > 1:
            noisy_partial_labels = labeled_set.partial.copy()

            # if label changed from 0 to 1, set all partial labels to 1
            noisy_partial_labels[np.logical_and(labeled_set.labels == 0, noisy_labels == 1)] = 1

            # if label changed from 1 to 0, select a random subspace to flip label
            mask = np.logical_and(labeled_set.labels == 1, noisy_labels == 0)
            noisy_partial_labels[mask, np.random.randint(0, labeled_set.num_partitions, size=mask.sum())] = 0

        return LabeledSet(noisy_labels, noisy_partial_labels, labeled_set.index.copy())

    return __skip_initial_points(injector, skip_initial)


def __skip_initial_points(noise_injector: NoiseInjector, skip_initial: int) -> NoiseInjector:
    """
    :param noise_injector: NoiseInjector to be decorated
    :param skip_initial: number of times to skip the noise injection
    :return: a decorated noise_injector, which will NOT add noise to the initial 'skip_initial' times it is called.
    """
    assert_non_negative_integer(skip_initial, 'skip_initial')

    def injector(iteration: int, labeled_set: LabeledSet) -> LabeledSet:
        return noise_injector(iteration, labeled_set) if iteration > skip_initial else labeled_set

    return injector


def __flip(labels, noise):
    flip_mask = np.random.rand(*labels.shape) < noise
    return np.where(flip_mask, 1 - labels, labels)
