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
