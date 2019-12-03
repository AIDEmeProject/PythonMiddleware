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
        :param switch_iter:
        :param text_column_start: column index where
        """
        self._al_struct = al_struct
        self._al_text = al_text
        self.__switch_iter = assert_positive_integer(switch_iter, 'switch_iter')
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

    def fit(self, X, y):
        """
        :param X: data matrix. We assume that all text columns are located at the end
        :param y: labels matrix. Last column are assumed to be the text labels
        :return:
        """
        self.__iteration = len(X)  # TODO: not exactly precise because of initial sampling...

        if self.is_struct_phase:
            X_struct, y_struct = self.get_struct_data(X), self.get_struct_labels(y)
            self._al_struct.fit(X_struct, y_struct)

        else:
            X_text, y_text = self.get_text_data(X), self.get_text_labels(y)
            self._al_text.fit(X_text, y_text)

    def predict(self, X):
        if self.is_text_phase and self.__mask_cache is None:
            self.__labels_cache = self._al_struct.predict(self.get_struct_data(X))
            self.__mask_cache = (self.__labels_cache == 1)
            self.__X_text = self.get_text_data(X[self.__mask_cache])

        return self.__compute(X, self._al_struct.predict, self._al_text.predict, self.__labels_cache)

    def predict_proba(self, X):
        if self.is_text_phase and self.__probas_cache is None:
            self.__probas_cache = self._al_struct.predict_proba(self.get_struct_data(X))

        return self.__compute(X, self._al_struct.predict_proba, self._al_text.predict_proba, self.__probas_cache)

    def __compute(self, X, f_struct, f_text, cache):
        if self.is_struct_phase:
            return f_struct(self.get_struct_data(X))

        values = cache.copy()
        values[self.__mask_cache] = f_text(self.__X_text)
        return values

    def rank(self, X):
        if self.is_struct_phase:
            return self._al_struct.rank(self.get_struct_data(X))

        return self._al_text.rank(self.__X_text)

    def next_points_to_label(self, data, subsample=None):
        if self.is_struct_phase:
            col_slice = slice(0, self.__text_column_start)
            return self._al_struct.next_points_to_label(data.select_cols(col_slice), subsample)

        return self._al_text.next_points_to_label(self.__X_text, subsample)
