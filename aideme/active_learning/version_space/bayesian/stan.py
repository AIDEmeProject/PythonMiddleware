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

import logging
import os
import pickle
from typing import TYPE_CHECKING

from pystan import StanModel

from aideme.utils import assert_positive_integer, assert_positive

if TYPE_CHECKING:
    import numpy as np


class StanLogisticRegressionSampler:
    """
    Logistic regression posterior sampler. Uses pystan library, which is basically a wrapper over the Bayesian modeling
    library STAN in R.

    Link: https://github.com/stan-dev/pystan
    """
    __stan_model = """
                   data {{
                       int<lower=1> D; 
                       int<lower=0> N; 
                       int<lower=0,upper=1> y[N]; 
                       matrix[N,D] x; 
                       real<lower=0> sigma;
                   }} 
                   parameters {{ 
                        vector[D] beta; 
                   }} 
                   model {{ 
                       {prior}
                       y ~ bernoulli_logit(x * beta); 
                   }}"""

    def __init__(self, warmup: int = 100, thin: int = 1, prior: str = 'improper', prior_std: float = 1.0,
                 suppress_warnings: bool = True):
        """
        :param warmup: number of initial samples to ignore
        :param thin: number of samples to skip
        :param prior: prior for logistic regression weights. Available options are: 'gaussian', 'cauchy', and 'improper'
        :param prior_std: standard deviation of prior distribution. It has no effect for 'improper' prior.
        :param suppress_warnings: whether to suppress all pystan warning log messages
        """
        assert_positive_integer(warmup, 'warmup')
        assert_positive_integer(thin, 'thin')
        assert_positive(prior_std, 'prior_std')

        self.warmup = warmup
        self.thin = thin

        self.model = self.__get_pystan_model(prior)
        self.prior_std = prior_std

        self.__suppress_pystan_logs(suppress_warnings)

    @staticmethod
    def __suppress_pystan_logs(suppress_warnings: bool) -> None:
        level = logging.ERROR if suppress_warnings else logging.INFO
        logging.getLogger('pystan').setLevel(level)

    def __get_pystan_model(self, prior: str) -> StanModel:
        path_to_saved_model = 'stan_models/{}.pkl'.format(prior)

        if os.path.isfile(path_to_saved_model):
            return pickle.load(open(path_to_saved_model, 'rb'))

        model = StanModel(model_code=self.__parse_prior_options(prior))
        pickle.dump(model, open(path_to_saved_model, 'wb'))
        return model

    def __parse_prior_options(self, prior: str):
        if prior == 'gaussian':
            prior_str = 'beta ~ normal(0, sigma);'

        elif prior == 'cauchy':
            prior_str = 'beta ~ cauchy(0, sigma);'

        elif prior == 'improper':
            prior_str = ''

        else:
            raise ValueError("Unknown prior: {}. Available options are: 'gaussian', 'cauchy', and 'improper'.".format(prior))

        model_code = self.__stan_model.format(prior=prior_str)

        return model_code

    def clear(self) -> None:
        pass

    def sample(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample from the posterior distribution through an MCMC sampler.

        :param X: data matrix
        :param y: labels (positive label should be 1)
        :param n_samples: number of samples
        :return: samples in a numpy array (one per line)
        """
        y = (y == 1).astype('int')

        data = {
            'N': X.shape[0],
            'D': X.shape[1],
            'x': X,
            'y': y,
            'sigma': self.prior_std
        }

        iter = self.warmup + self.thin * n_samples
        result = self.model.sampling(data=data, iter=iter, warmup=self.warmup, thin=self.thin, chains=1)
        return result.extract()['beta']
