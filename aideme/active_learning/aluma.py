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
from math import ceil
from typing import Optional, Union

import numpy as np
import scipy
import scipy.special

from aideme.active_learning.kernel import Kernel
from aideme.utils import assert_non_negative, assert_in_range, assert_positive
from aideme.utils.random import get_random_state


def aluma_preprocessing(X: np.ndarray, kernel: Optional[Kernel], margin: float, H: float, delta: float,
                        seed: Optional[Union[int, np.random.RandomState]] = None) -> np.ndarray:
    """
    :param X: data matrix
    :param kernel: kernel function. If None, no kernel will be applied
    :param margin: estimate of best separator's margin
    :param H: upper bound on hinge loss of best separator, i.e., there is 'w' s.t. H >= sum_i max(0, margin - y_i w^T x_i)
    :param delta: probability of keeping linear separability
    :return: the pre-processes matrix
    """
    assert_non_negative(H, 'H')
    assert_positive(margin, 'margin')
    assert_in_range(delta, 'delta', 0, 1)

    rng = get_random_state(seed)

    eps = 0.5 * margin / (1 + np.sqrt(H))
    k = ceil(4 * np.log(4 * X.shape[0] / delta) / (eps * eps * (1 - eps)))

    if kernel:
        K = kernel(X)
        scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition
    else:
        K = X.copy()

    a = 1 / np.sqrt(1 + np.sqrt(H))
    sq_a = np.sqrt(1 - a * a)

    K = K @ np.where(rng.rand(K.shape[1], k) > 0.5, a, -a)
    K += np.where(rng.rand(K.shape[0], k) > 0.5, sq_a, -sq_a)

    return K
