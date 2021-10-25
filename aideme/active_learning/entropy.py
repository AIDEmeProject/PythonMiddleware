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
from scipy.special import xlogy

from .version_space import KernelVersionSpace
from ..utils import assert_positive_integer, metric_logger


class EntropyReductionLearner(KernelVersionSpace):

    def __init__(self, data_sample_size: Optional[int] = None,
                 single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, rounding_cache: bool = True, rounding_options: Optional[Dict] = None,
                 add_intercept: bool = True, decompose: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):

        assert_positive_integer(data_sample_size, 'data_sample_size', allow_none=True)

        super().__init__(
            single_chain=single_chain, n_samples=n_samples, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding, rounding_cache=rounding_cache, rounding_options=rounding_options,
            add_intercept=add_intercept, decompose=decompose,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        self.data_sample_size = data_sample_size
        self.hypothesis_sample_size = n_samples
        self.__sample = None

    def clear(self) -> None:
        super().clear()
        self.__sample = None

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_all(X)

    def fit_data(self, data) -> None:
        super().fit_data(data)
        self.__sample = data.sample(self.data_sample_size)

    def rank(self, X: np.ndarray) -> np.ndarray:
        ranks = super().rank(X)

        min_rank = ranks.min()
        metric_logger.log_metric('min_cut_proba', min_rank)

        if min_rank == 0.5:  # no point cuts the VS
            return np.zeros(len(X))

        # among the points reducing the VS by the most, select the one with smallest expected prediction entropy after labeling
        is_rank_minimizer = (ranks == min_rank)
        ranks[~is_rank_minimizer] = np.inf
        ranks[is_rank_minimizer] = self.__entropy(X[is_rank_minimizer])

        return ranks

    @metric_logger.log_execution_time('entropy_time')
    def __entropy(self, X: np.ndarray) -> np.ndarray:
        H = self.predict_all(X)  # samples x data points
        H_sample = self.predict_all(self.__sample)

        sums = H.sum(axis=0)
        cut_proba = sums / H.shape[0]

        pos_probas = H.T @ H_sample
        neg_probas = H_sample.sum(axis=0) - pos_probas

        pos_probas /= sums.reshape(-1, 1)
        neg_probas /= (H.shape[0] - sums).reshape(-1, 1)

        pos_entropy = self.__compute_average_entropy(pos_probas)
        neg_entropy = self.__compute_average_entropy(neg_probas)

        return pos_entropy * cut_proba + neg_entropy * (1 - cut_proba)

    @staticmethod
    def __compute_average_entropy(p: np.ndarray) -> np.ndarray:
        mp = 1 - p
        res = -xlogy(p, p).mean(axis=-1)
        res -= xlogy(mp, mp).mean(axis=-1)
        return res
