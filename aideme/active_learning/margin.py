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

"""
SVM-based Active Learning algorithms, from the article "Support Vector Machine Active Learning with Applications to Text
Classification", JMLR (2001), by Simon TONG & Daphne KOLLER.

Link: http://www.jmlr.org/papers/volume2/tong01a/tong01a.pdf
"""

from typing import Union

import numpy as np
from scipy.special import expit
from sklearn import clone
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .uncertainty import UncertaintySampler


class SimpleMargin(UncertaintySampler):
    """
    At every iteration, it trains an SVM model over labeled data, and picks the closest point to the decision boundary
    as most informative point.
    """
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: Union[str, float]='auto'):
        clf = SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape='ovo')
        super().__init__(clf)

    def predict_proba(self, X):
        return expit(self.clf.decision_function(X))

    def rank(self, X):
        """
        Rank points based on their distance to the decision boundary.
        :param X: data matrix
        :return: distance list
        """
        return np.abs(self.clf.decision_function(X))


class RatioMargin(SimpleMargin):
    """
    For every unlabeled point, we train two SVM models: one positively biased and one negatively biased. The SVM's margin
    is an estimate for the Version Space remaining volume; thus, the point that most closely halves the current version
    space is the one whose two previously computed margins are the closest to each other.
    """
    def fit(self, X, y):
        self.clf.fit(X, y)

        # store training data
        self.X_train = np.vstack([X, np.zeros(X.shape[1])])
        self.y_train = np.hstack([y, 0])

    def rank(self, X):
        # check model is fitted
        check_is_fitted(self.clf, 'support_')

        # clone fitted model to avoid losing its weights
        clf = clone(self.clf)

        # add "-" sign because we want the LARGEST margin to be returned
        return -np.array([self.__compute_margin_ratio(clf, x) for x in X])

    def __compute_margin_ratio(self, clf, x):
        margin0 = self.__compute_margin(clf, x, clf.classes_[0])
        margin1 = self.__compute_margin(clf, x, clf.classes_[1])

        if margin0 <= 0 or margin1 <= 0:
            return float('inf')

        return min(margin1 / margin0, margin0 / margin1)

    def __compute_margin(self, clf, x, y):
        # set training data
        self.X_train[-1] = x
        self.y_train[-1] = y

        # train classifier and
        clf.fit(self.X_train, self.y_train)
        return float(clf.dual_coef_.dot(clf.decision_function(clf.support_vectors_)))
