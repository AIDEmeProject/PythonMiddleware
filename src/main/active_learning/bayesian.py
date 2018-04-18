import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel

from .base import ActiveLearner
from pystan import StanModel

class StanLogregSampler:
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

    def __init__(self, add_intercept=False, iter=1000, warmup=500, thin=1, sigma=1.):
        # add intercept to linear model?
        self.add_intercept = add_intercept

        # MCMC parameters
        self.iter = iter
        self.warmup = warmup
        self.thin = thin

        # gaussian prior standar deviation
        self.sigma = sigma

        # STAN model
        self.model = StanModel(model_code=self.__stan_model)

    def __preprocess(self, X, Y):
        """
        Add ones column to X (if necessary), and cast Y to {0,1} integer array
        """
        if self.add_intercept:
            ones = np.ones((len(X), 1))
            X = np.hstack([ones, X])
        Y = np.array(Y == 1, dtype='int')
        return X, Y

    def sample(self, X, Y):
        """
        Sample from the posterior distribution given the data (X,Y)
        """
        X, Y = self.__preprocess(X, Y)
        data_dict = {
            'N': X.shape[0],
            'D': X.shape[1],
            'x': X,
            'y': Y,
            'sig0': self.sigma
        }

        result = self.model.sampling(data=data_dict, iter=self.iter, warmup=self.warmup, thin=self.thin, chains=1)
        return result.extract()['beta']


class BayesianLogisticActiveLearner(ActiveLearner):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        self.samples = None

    def clear(self):
        self.samples = None

    def predict(self, X):
        return 2. * (self.predict_proba(X) > 0.5) - 1.

    def predict_proba(self, X):
        if X.shape[1] + 1 == self.samples.shape[1]:
            bias, weights = self.samples[:, 0].reshape(-1, 1), self.samples[:, 1:]
        else:
            bias, weights = 0, self.samples
        return np.mean(expit(bias + weights.dot(np.transpose(X))), axis=0)

    def fit_classifier(self, X, y):
        self.samples = self.sampler.sample(X, y)

    def ranker(self, data):
        return (self.predict_proba(data) - 0.5)**2


class KernelBayesianActiveLearner(ActiveLearner):
    def __init__(self, sampler, gamma=None):
        super().__init__()
        self.linear_learner = BayesianLogisticActiveLearner(sampler)
        self.X = None
        self.gamma = gamma

    def clear(self):
        self.X = None
        self.linear_learner.clear()

    def __preprocess(self, X):
        return rbf_kernel(X, self.X, gamma=self.gamma)

    def predict(self, X):
        return self.linear_learner.predict(self.__preprocess(X))

    def predict_proba(self, X):
        return self.linear_learner.predict_proba(self.__preprocess(X))

    def fit_classifier(self, X, y):
        X, y = np.array(X), np.array(y)
        self.X = X.copy()
        self.linear_learner.fit_classifier(self.__preprocess(X), y)

    def ranker(self, data):
        return self.linear_learner.ranker(self.__preprocess(data))
