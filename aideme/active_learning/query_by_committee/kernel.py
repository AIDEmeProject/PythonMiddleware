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

from functools import partial

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .linear import BayesianLogisticRegression


class KernelLogisticRegression:
    """
    Add kernel support to LinearBayesianLogisticRegression classifier. Basically, the data matrix X is substituted by
    the Kernel matrix K, depending on the chosen kernel ('linear', 'rbf', 'poly', or user-defined).
    """

    def __init__(self, n_samples=8, add_intercept=True, sampling='deterministic', warmup=100, thin=1, sigma=100.0,
                 cache=True, rounding=True, max_rounding_iters=None, strategy='default', z_cut=False,
                 kernel='rbf', gamma=None, degree=3, coef0=0.):
        self.logreg = BayesianLogisticRegression(sampling=sampling, n_samples=n_samples, warmup=warmup, thin=thin, sigma=sigma,
                                                 cache=cache, rounding=rounding, max_rounding_iters=max_rounding_iters,
                                                 strategy=strategy, z_cut=z_cut, add_intercept=add_intercept)
        self.kernel = self.__get_kernel(kernel, gamma, degree, coef0)

    @staticmethod
    def __get_kernel(kernel, gamma, degree, coef0):
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
        return self.kernel(X, self.X_train)

    def fit(self, X, y):
        self.X_train = check_array(X, copy=True)
        self.logreg.fit(self.__preprocess(X), y)

    def predict(self, X):
        check_is_fitted(self, 'X_train')
        return self.logreg.predict(self.__preprocess(X))

    def predict_proba(self, X):
        check_is_fitted(self, 'X_train')
        return self.logreg.predict_proba(self.__preprocess(X))
