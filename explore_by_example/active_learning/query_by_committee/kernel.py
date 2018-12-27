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

    def __init__(self, n_samples, add_intercept=True, sampling='bayesian', warmup=100, thin=1, sigma=100.0, rounding=True,
                       kernel='linear', gamma=None, degree=3, coef0=0.):
        self.logreg = BayesianLogisticRegression(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling,
                                                 warmup=warmup, thin=thin, sigma=sigma, rounding=rounding)
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
