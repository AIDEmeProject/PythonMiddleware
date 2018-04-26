from functools import partial
import pickle
import os

import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from pystan import StanModel

from definitions import RESOURCES_DIR
from .active_learner import ActiveLearner


class LogregSampler:
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
             beta ~ normal(0,sig0); 
             y ~ bernoulli_logit(x * beta); 
        } 
    """

    def __init__(self, add_intercept=True, iter=1000, warmup=500, thin=1, sigma=1.):
        # add intercept to linear model?
        self.add_intercept = add_intercept

        # MCMC parameters
        self.iter = iter
        self.warmup = warmup
        self.thin = thin

        # gaussian prior standard deviation
        self.sigma = sigma

        # STAN model
        model_path = os.path.join(RESOURCES_DIR, 'stan_logreg.pkl')
        if os.path.isfile(model_path):
            self.model = pickle.load(open(model_path, 'rb'))
        else:
            self.model = StanModel(model_code=self.__stan_model)
            pickle.dump(self.model, open(model_path, 'wb'))

    def __preprocess(self, X, y):
        """
        Add ones column to X (if necessary), and cast Y to {0,1} integer array
        """
        if self.add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])
        y = np.array(y == 1, dtype='int')
        return X, y

    def sample(self, X, y):
        """
        Sample from the posterior distribution given the data (X,Y)
        """
        X, y = self.__preprocess(X, y)
        data = {
            'N': X.shape[0],
            'D': X.shape[1],
            'x': X,
            'y': y,
            'sig0': self.sigma
        }

        result = self.model.sampling(data=data, iter=self.iter, warmup=self.warmup, thin=self.thin, chains=1)
        return result.extract()['beta']


class LinearBayesianQueryByCommittee(ActiveLearner):
    def __init__(self, sampler):
        ActiveLearner.__init__(self)
        self.sampler = sampler
        self.samples = None

    def fit(self, X, y):
        self.samples = self.sampler.sample(X, y)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X):
        if X.shape[1] + 1 == self.samples.shape[1]:
            bias, weights = self.samples[:, 0].reshape(-1, 1), self.samples[:, 1:]
        else:
            bias, weights = 0, self.samples
        return np.mean(expit(bias + weights.dot(X.T)), axis=0)

    def rank(self, X):
        return np.abs(self.predict_proba(X) - 0.5)


class KernelBayesianQueryByCommittee(ActiveLearner):

    def __init__(self, sampler, kernel='linear', gamma=None, degree=3, coef0=0.):
        ActiveLearner.__init__(self)
        self.linear_learner = LinearBayesianQueryByCommittee(sampler)
        self.kernel = self.__get_kernel(kernel, gamma, degree, coef0)
        self.X = None

    @staticmethod
    def __get_kernel(kernel, gamma=None, degree=3, coef0=0.):
        if kernel == 'linear':
            return linear_kernel
        elif kernel == 'poly':
            return partial(polynomial_kernel, gamma=gamma, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            return partial(rbf_kernel, gamma=gamma)
        elif callable(kernel):
            return kernel

        raise ValueError("Unsupported kernel. Available options are 'linear', 'rbf', 'poly', or any custom K(X,Y) function.")

    def __preprocess(self, X):
        return self.kernel(X, self.X)

    def fit(self, X, y):
        self.X = X.copy()
        self.linear_learner.fit(self.__preprocess(X), y)

    def predict(self, X):
        return self.linear_learner.predict(self.__preprocess(X))

    def predict_proba(self, X):
        return self.linear_learner.predict_proba(self.__preprocess(X))

    def rank(self, X):
        return self.linear_learner.rank(self.__preprocess(X))
