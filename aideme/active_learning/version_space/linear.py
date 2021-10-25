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

import numpy as np

from .hit_and_run import HitAndRunSampler


class BayesianLogisticRegressionBase:
    def __init__(self, sampler,  n_samples: int = 8, add_intercept: bool = True):
        """
        :param sampler: sampling method
        :param n_samples: number of samples to compute from posterior
        :param add_intercept: whether to add an intercept or not
        """
        self.sampler = sampler
        self.n_samples = n_samples
        self.add_intercept = add_intercept

    def clear(self) -> None:
        self.sampler.clear()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.add_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        samples = self.sampler.sample(X, y, self.n_samples)

        if self.add_intercept:
            self.bias, self.weight = samples[:, 0].reshape(-1, 1), samples[:, 1:]
        else:
            self.bias, self.weight = 0, samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean(self._likelihood(X), axis=0)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _margin(self, X: np.ndarray) -> np.ndarray:
        return self.bias + self.weight @ X.T


class DeterministicLogisticRegression(BayesianLogisticRegressionBase):
    """
    DETERMINISTIC POSTERIOR
        p(w | X, y) = 1 if y_i X_i^T w > 0 for all i, else 0

    In this particular case, we assume no labeling noise, and data should be linear separable. However, we can achieve
    better performance under these assumptions.
    """

    def __init__(self, single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10,
                 cache_samples: bool = True, rounding: bool = True,  rounding_cache: bool = True,
                 rounding_options: Optional[Dict] = None, add_intercept: bool = True):
        """
        :param n_samples: number of samples to compute from posterior
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param cache_samples: whether to cache previous samples in order to speed-up 'initial point' computation in hit-and-run.
        :param rounding: whether to apply a rounding procedure in the 'deterministic' sampling.
        :param rounding_cache: whether cache rounding ellipsoid between iterations. Significantly speeds-up computations, but performance may suffer a little.
        :param rounding_options: dictionary containing the rounding algorithm configuration. See RoundingAlgorithm class
        for possible values are defaults.
        :param add_intercept: whether to add an intercept or not
        """
        sampler = HitAndRunSampler(
            single_chain=single_chain, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding, rounding_cache=rounding_cache, rounding_options=rounding_options
        )

        super().__init__(sampler=sampler, n_samples=n_samples, add_intercept=add_intercept)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        return (self._margin(X) > 0).astype('float')
