import random
import warnings

import numpy as np

from .factorization import FactorizedPolytopeModel
from .persistent import PolytopeModel
from ..active_learner import FactorizedActiveLearner


class DualSpaceModel(FactorizedActiveLearner):
    """
    Dual Space model
    """

    def __init__(self, active_learner, sample_unknown_proba=0.5, partition=None, mode='persist', tol=1e-12):
        self.active_learner = active_learner
        self.sample_unknown_proba = sample_unknown_proba
        self.__partition = partition
        self.__mode = mode
        self.__tol = tol
        self.set_factorization_structure()

    def clear(self):
        self.active_learner.clear()
        self.polytope_model.clear()

    def set_factorization_structure(self, **factorization_info):
        partition = factorization_info.get('partition', self.__partition)
        mode = factorization_info.get('mode', self.__mode)

        if not partition:
            self.polytope_model = PolytopeModel(mode, self.__tol)
            self.factorized = False

        else:
            size = len(partition)

            if isinstance(mode, str):
                mode = [mode] * size

            if len(mode) != size:
                raise ValueError("'mode' and 'partition' parameters have incompatible lengths.")

            self.polytope_model = FactorizedPolytopeModel(partition, mode, self.__tol)
            self.factorized = True

    def fit_data(self, data):
        """
        Fits both active learner and polytope model.
        """
        self.__fit_active_learner(data)

        if not self.polytope_model.is_valid:
            return

        X_new, y_new = data.last_labeled_set
        is_success = self.polytope_model.update(X_new, y_new)

        # if conflicting points were found, the inferred partition has to be relabeled and labeled points checked
        if not is_success:
            warnings.warn("Found conflicting point in polytope model. is_valid = {0}".format(self.polytope_model.is_valid))
            data.remove_inferred()
            self.__fit_active_learner(data)  # retrain AL since labeled set may have changed

            # if polytope became invalid with the last update, skip relabeling
            if not self.polytope_model.is_valid:
                return

        unk_idx, unk = data.unknown
        pred = self.polytope_model.predict(unk)
        data.move_to_inferred(unk_idx[pred != 0.5])

    def predict(self, X):
        """
        Predicts classes based on polytope model first; unknown labels are labeled via the active learner
        """
        if not self.polytope_model.is_valid:
            return self.active_learner.predict(X)

        predictions = self.polytope_model.predict(X)

        unknown_mask = (predictions == 0.5)

        if np.any(unknown_mask):
            predictions[unknown_mask] = self.active_learner.predict(X[unknown_mask])

        return predictions

    def predict_proba(self, X):
        """
        Predicts probabilities using the polytope model first; unknown labels are predicted via the active learner
        """
        if not self.polytope_model.is_valid:
            return self.active_learner.predict_proba(X)

        probas = self.polytope_model.predict(X)

        unknown_mask = (probas == 0.5)

        if np.any(unknown_mask):
            probas[unknown_mask] = self.active_learner.predict_proba(X[unknown_mask])

        return probas

    def rank(self, X):
        """
        Simply use AL to rank points
        """
        return self.active_learner.rank(X)

    def next_points_to_label(self, data, subsample=None):
        if not self.polytope_model.is_valid:
            return self.active_learner.next_points_to_label(data=data, subsample=subsample)

        while data.unknown_size > 0:
            idx_sample, X_sample = data.sample_unknown(subsample) if random.random() < self.sample_unknown_proba else data.sample_unlabeled(subsample)
            idx_selected, X_selected = self.active_learner._select_next(idx_sample, X_sample)

            if self.factorized:
                pred = self.polytope_model.predict_partial(X_selected)
                is_known = (np.min(pred, axis=1) != 0.5)
            else:
                pred = self.polytope_model.predict(X_selected)
                is_known = (pred != 0.5)

            if np.any(is_known):
                data.move_to_labeled([idx for i, idx in enumerate(idx_selected) if is_known[i]], pred[is_known], 'dsm')
                self.__fit_active_learner(data)

            if not np.all(is_known):
                return [idx for i, idx in enumerate(idx_selected) if not is_known[i]], X_selected[~is_known]

        return self.active_learner.next_points_to_label(data=data, subsample=subsample)

    def __fit_active_learner(self, data):
        X, y = data.training_set

        if self.factorized:
            y = np.min(y, axis=1)

        self.active_learner.fit(X, y)
