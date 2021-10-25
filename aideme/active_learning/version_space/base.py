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
from typing import Optional, Dict

from .kernel import KernelBayesianLogisticRegression
from .linear import DeterministicLogisticRegression
from ..uncertainty import UncertaintySampler


class VersionSpaceBase(UncertaintySampler):
    def __init__(self, logreg):
        UncertaintySampler.__init__(self, logreg)

    def clear(self) -> None:
        self.clf.clear()


class LinearVersionSpace(VersionSpaceBase):
    def __init__(self, single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10,
                 cache_samples: bool = True, rounding: bool = True, rounding_cache: bool = True,
                 rounding_options: Optional[Dict] = None, add_intercept: bool = True):
        logreg = DeterministicLogisticRegression(
            single_chain=single_chain, n_samples=n_samples, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding, rounding_cache=rounding_cache, rounding_options=rounding_options,
            add_intercept=add_intercept
        )

        super().__init__(logreg)


class KernelVersionSpace(VersionSpaceBase):
    def __init__(self, single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, rounding_cache: bool = True, rounding_options: Optional[Dict] = None,
                 add_intercept: bool = True, decompose: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        if not rounding:
            rounding_cache = False

        if rounding_cache:
            decompose = True

        logreg = DeterministicLogisticRegression(
            single_chain=single_chain, n_samples=n_samples, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding,  rounding_cache=rounding_cache, rounding_options=rounding_options,
            add_intercept=add_intercept
        )

        kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=decompose,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        super().__init__(kernel_logreg)
