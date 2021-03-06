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

from .active_learner import ActiveLearner


class UncertaintySampler(ActiveLearner):
    """
    Uncertainty sampling is the most simple and popular Active Learning class of algorithms. Basically, given any classifier
    (Random Forest, SVM, ...) it will rank unlabeled points based one the estimated class probabilities: the closest to 0.5,
    the most uncertain (hence more informative) the sample is.
    """
    def __init__(self, clf):
        """
        :param clf: classifier object implementing 3 methods:
            - fit(X, y): trains a classification model over the labeled data
            - predict(X): predicts class label for each row of matrix X
            - predict_proba(X): predicts probability of being positive for each row of matrix X

            It should be compatible with most Scikit-learn library's classifiers.
        """
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def rank(self, X):
        return np.abs(self.predict_proba(X) - 0.5)
