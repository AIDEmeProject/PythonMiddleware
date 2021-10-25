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
from .linear import StanBayesianLogisticRegression
from .laplace import LaplaceBayesianLogisticRegression, KernelLaplaceBayesianLogisticRegression
from ..base import VersionSpaceBase
from ..kernel import KernelBayesianLogisticRegression


class BayesianLinearVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'laplace', prior: str = 'improper', prior_std: float = 1.0,
                 tol: float = 1e-6, max_iter: int = 10000, suppress_warnings: bool = True):
        if sampler == 'stan':
            logreg = StanBayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'laplace':
            logreg = LaplaceBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept, tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Unknown sampler option: {}".format(sampler))

        super().__init__(logreg)


class BayesianKernelVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'laplace', prior: str = 'improper', prior_std: float = 1.0,
                 tol: float = 1e-6, max_iter: int = 10000, suppress_warnings: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        if sampler == 'stan':
            logreg = StanBayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'laplace':
            logreg = LaplaceBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept, tol=tol, max_iter=max_iter)
        elif sampler == 'kernel-laplace':
            logreg = KernelLaplaceBayesianLogisticRegression(prior_std=prior_std, tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Unknown sampler option: {}. Available options are: 'stan', 'laplace', and 'kernel-laplace'.".format(sampler))

        kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=False,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        super().__init__(kernel_logreg)
