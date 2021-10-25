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

from typing import Optional, Union, List, TYPE_CHECKING

import numpy as np

from .linear import LinearFactorizationLearner
from .optimization import OptimizationAlgorithm
from ..kernel import KernelTransformer, FactorizedKernelTransform

if TYPE_CHECKING:
    from .penalty import PenaltyTerm


class KernelFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, penalty_term: Optional[PenaltyTerm] = None, add_bias: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12, nystroem_components: Optional[int] = None):
        self.fact_linear = LinearFactorizationLearner(optimizer=optimizer, penalty_term=penalty_term, add_bias=add_bias)
        self._base_kernel_transform = KernelTransformer.get(kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter, nystroem_components=nystroem_components)
        self._kernel_transformer = None
        self._factorization = None

    @property
    def bias(self) -> np.ndarray:
        return self.fact_linear.bias

    @property
    def weights(self) -> np.ndarray:
        return self.fact_linear.weights

    @property
    def weight_matrix(self) -> Optional[np.ndarray]:
        return self.fact_linear.weight_matrix

    @property
    def num_subspaces(self) -> int:
        return self.fact_linear.num_subspaces

    def _get_loss(self, X: np.ndarray, y: np.ndarray):
        return self.fact_linear._get_loss(self._get_kernel_matrix(X), y)

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], retries: int = 1, x0: Optional[np.ndarray] = None):
        if isinstance(factorization, int):
            self._factorization = None
        else:
            self._factorization = factorization
            factorization = len(factorization)

        self._kernel_transformer = self._get_kernel_transform()
        K = self._kernel_transformer.fit_transform(X)

        return self.fact_linear.fit(K, y, factorization, retries, x0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict(self._get_kernel_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict_proba(self._get_kernel_matrix(X))

    def partial_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.partial_proba(self._get_kernel_matrix(X))

    def _get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._kernel_transformer.transform(X)

    def _get_kernel_transform(self) -> KernelTransformer:
        if self._factorization is None:
            return self._base_kernel_transform.clone()

        return FactorizedKernelTransform(self._base_kernel_transform, self._factorization)
