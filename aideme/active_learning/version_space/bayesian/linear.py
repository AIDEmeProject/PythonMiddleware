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
import numpy as np
from scipy.special import expit

from .stan import StanLogisticRegressionSampler
from ..linear import BayesianLogisticRegressionBase


class StanBayesianLogisticRegression(BayesianLogisticRegressionBase):
    """
    LOGISTIC POSTERIOR
        p(w | X, y) ~= exp(-|w|^2 / sigma^2) / \prod_i (1 + exp(-y_i X_i^T w))

    Basically, we have chosen a centered gaussian prior and a logistic likelihood function.
    """

    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 prior: str = 'improper', prior_std: float = 1.0, suppress_warnings: bool = True):
        """
        :param n_samples: number of samples to compute from posterior
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param add_intercept: whether to add an intercept or not
        :param prior: prior for logistic regression weights. Available options are: 'gaussian', 'cauchy', and 'improper'
        :param prior_std: standard deviation of prior distribution. It has no effect for 'improper' prior.
        :param suppress_warnings: whether to suppress all pystan warning log messages
        """
        sampler = StanLogisticRegressionSampler(warmup=warmup, thin=thin, prior=prior, prior_std=prior_std,
                                                suppress_warnings=suppress_warnings, )

        super().__init__(sampler=sampler, n_samples=n_samples, add_intercept=add_intercept)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        return expit(self._margin(X))
