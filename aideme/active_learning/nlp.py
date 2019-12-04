import numpy as np

from aideme.active_learning.active_learner import FactorizedActiveLearner
from aideme.utils import assert_positive_integer


class TwoStepsLearner(FactorizedActiveLearner):
    """
    A special Active Learner for working with datasets containing both structured and text data. It works in two phases:

        Phase 1: Run an Active Learner of choice over structured data only

        Phase 2: Run another Active Learner of choice over the text data, but filtering it to the points where the first
        Active Learner predicts as positive.
    """

    def __init__(self, al_struct, al_text, switch_iter, text_column_start):
        """
        :param al_struct: Active Learner to run over structured data
        :param al_text: Active Learner to run over text data
        :param switch_iter: iteration to switch from structured data exploration to text data exploration
        :param text_column_start: column index where
        """
        assert_positive_integer(switch_iter, 'switch_iter')

        self._al_struct = al_struct
        self._al_text = al_text
        self.__switch_iter = switch_iter
        self.__text_column_start = text_column_start
        self.__iteration = 0
        self.__labels_cache = None
        self.__probas_cache = None
        self.__mask_cache = None
        self.__X_text = None

    def clear(self):
        self._al_struct.clear()
        self._al_text.clear()
        self.__iteration = 0
        self.__labels_cache = None
        self.__probas_cache = None
        self.__mask_cache = None
        self.__X_text = None

    @property
    def is_struct_phase(self):
        return self.__iteration < self.__switch_iter

    @property
    def is_text_phase(self):
        return not self.is_struct_phase

    def get_struct_data(self, X):
        return X[:, :self.__text_column_start]

    def get_struct_labels(self, y):
        return y[:, :-1]

    def get_text_data(self, X):
        return X[:, self.__text_column_start:]

    def get_text_labels(self, y):
        return y[:, -1]

    def fit_data(self, data):
        self.__iteration = data.labeled_size

        if self.is_struct_phase:
            col_slice = slice(0, self.__text_column_start)
            lb_slice = slice(0, -1)
            self._al_struct.fit_data(data.select_cols(col_slice, lb_slice))

        else:
            if self.__labels_cache is None:
                self.__update_cache(data.data)

                mask = self.__mask_cache.copy()
                mask[data.labeled[0]] = False  # remove labeled points
                idx = np.arange(len(data.data))[mask]
                data.move_to_inferred(idx)

            col_slice = slice(self.__text_column_start, None)
            lb_slice = -1
            self._al_text.fit_data(data.select_cols(col_slice, lb_slice))

    def predict(self, X):
        return self.__compute(X, self._al_struct.predict, self._al_text.predict, self.__labels_cache)

    def predict_proba(self, X):
        return self.__compute(X, self._al_struct.predict_proba, self._al_text.predict_proba, self.__probas_cache)

    def __compute(self, X, f_struct, f_text, cache):
        if self.is_struct_phase:
            return f_struct(self.get_struct_data(X))

        values = cache.copy()
        values[self.__mask_cache] = f_text(self.__X_text)
        return values

    def __update_cache(self, X):
        self.__labels_cache = self._al_struct.predict(self.get_struct_data(X))
        self.__probas_cache = self._al_struct.predict_proba(self.get_struct_data(X))
        self.__mask_cache = (self.__labels_cache == 1)
        self.__X_text = self.get_text_data(X[self.__mask_cache])

    def next_points_to_label(self, data, subsample=None):
        if self.is_struct_phase:
            col_slice = slice(0, self.__text_column_start)
            lb_slice = slice(0, -1)
            return self._al_struct.next_points_to_label(data.select_cols(col_slice, lb_slice), subsample)

        idx, X = data.sample_inferred(subsample)
        return self._al_text._select_next(idx, self.get_text_data(X))
