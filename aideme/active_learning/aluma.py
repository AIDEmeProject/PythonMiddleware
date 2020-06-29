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
from typing import Optional

import numpy as np
import scipy

from aideme.active_learning.kernel import Kernel
from aideme.utils.validation import assert_non_negative, assert_positive


def aluma_preprocessing(X: np.ndarray, kernel: Optional[Kernel], gamma: float, H: float, delta: float) -> np.ndarray:
    assert_non_negative(H, 'H')
    assert_positive(gamma, 'gamma')
    assert_positive(delta, 'delta')

    k = 4 * (H + 1) * np.log(X.shape[0] / delta) / (gamma * gamma)  # TODO: check multiplying factor

    K = X
    if kernel:
        K = kernel(X)
        scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition

    a = 1 / np.sqrt(1 + np.sqrt(H))
    sq_a = np.sqrt(1 - a*a)

    M1 = np.where(np.random.rand(K.shape[1], k) > 0.5, a, -a)
    M2 = np.where(np.random.rand(K.shape[0], k) > 0.5, sq_a, -sq_a)

    return K @ M1 + M2
