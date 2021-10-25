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
import sklearn

from .active_learner import ActiveLearner
from ..utils import assert_positive, assert_positive_integer


class QueryByDisagreement(ActiveLearner):

    def __init__(self, learner, background_sample_size: int = 200, background_sample_weight: float = 1e-5):
        assert_positive_integer(background_sample_size, 'background_sample_size')
        assert_positive(background_sample_weight, 'background_sample_weight')

        self._background_sample_size = background_sample_size
        self._background_sample_weight = background_sample_weight

        self._learner = learner
        self._positively_biased_learner = sklearn.base.clone(learner)
        self._negatively_biased_learner = sklearn.base.clone(learner)

    def fit_data(self, data) -> None:
        X, y = data.training_set()
        self._learner.fit(X, y)

        background_points = data.unlabeled.sample(self._background_sample_size).data
        X_train = np.r_[X, background_points]

        sample_weights = np.ones(len(X_train))
        sample_weights[-self._background_sample_size:] *= self._background_sample_weight

        y_train = np.r_[y, np.ones(self._background_sample_size)]
        self._positively_biased_learner.fit(X_train, y_train, sample_weights)

        y_train[-self._background_sample_size:] = 0
        self._negatively_biased_learner.fit(X_train, y_train, sample_weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._learner.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._learner.predict_proba(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        """ Pick a random point for which the positively and negatively biased classifiers differ. """

        positively_biased_labels = self._positively_biased_learner.predict(X)  # type: np.ndarray
        negatively_biased_labels = self._negatively_biased_learner.predict(X)  # type: np.ndarray

        return (positively_biased_labels == negatively_biased_labels).astype('float')
