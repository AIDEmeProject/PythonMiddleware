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
from __future__ import annotations

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
        data { 
             int<lower=1> D; 
             int<lower=0> N; 
             int<lower=0,upper=1> y[N]; 
             matrix[N,D] x; 
             real<lower=0> sig0;
        } 
        parameters { 
             vector[D] beta; 
        } 
        model { 
             beta ~ normal(0, sig0); 
             y ~ bernoulli_logit(x * beta); 
        } 
    """

    def __init__(self, warmup: int = 100, thin: int = 1,
                 sigma: float = 1.0,
                 suppress_warnings: bool = True):
        """
        :param warmup: number of initial samples to ignore
        :param thin: number of samples to skip
        :param sigma: Gaussian prior standard deviation
        :param suppress_warnings: whether to suppress all pystan warning log messages
        """
        assert_positive_integer(warmup, 'warmup')
        assert_positive_integer(thin, 'thin')
        assert_positive(sigma, 'sigma')

        self.warmup = warmup
        self.thin = thin

        self.__suppress_pystan_logs(suppress_warnings)
        self.model = self.__get_pystan_model()
        self.model_params = {'sig0': sigma}

    def __suppress_pystan_logs(self, suppress_warnings):
        import logging
        level = logging.ERROR if suppress_warnings else logging.INFO
        logging.getLogger('pystan').setLevel(level)

    def __get_pystan_model(self) -> StanModel:
        model_path = 'stan_logreg.pkl'

        if os.path.isfile(model_path):
            return pickle.load(open(model_path, 'rb'))

        model = StanModel(model_code=self.__stan_model)
        pickle.dump(model, open(model_path, 'wb'))
        return model

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
        }
        data.update(self.model_params)

        iter = self.warmup + self.thin * n_samples
        result = self.model.sampling(data=data, iter=iter, warmup=self.warmup, thin=self.thin, chains=1)
        return result.extract()['beta']
