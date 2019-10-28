import random

import numpy as np

from .polytope import PolytopeModel
from .factorization import FactorizedPolytopeModel
from ..active_learner import ActiveLearner


class DualSpaceModel(ActiveLearner):
    """
    Dual Space model
    """

    def __init__(self, active_learner, sample_unknown_proba=0.5, partition=None, seed=None, tol=1e-12):
        self.active_learner = active_learner
        self.sample_unknown_proba = sample_unknown_proba
        self.polytope_model = PolytopeModel(tol) if partition is None else FactorizedPolytopeModel(partition, tol)
        self.factorized = partition is not None
        self.rng = random.Random(seed)

    def clear(self):
        self.active_learner.clear()
        self.polytope_model.clear()

    def fit_data(self, data):
        """
        Fits both active learner and polytope model.
        """
        self.__fit_active_learner(data)

        X_new, y_new = data.last_labeled_set
        self.polytope_model.update(X_new, y_new)

        unk_idx, unk = data.unknown
        pred = self.polytope_model.predict(unk)
        data.move_to_inferred(unk_idx[pred != 0.5])

    def predict(self, X):
        """
        Predicts classes based on polytope model first; unknown labels are labeled via the active learner
        """
        predictions = self.polytope_model.predict(X)

        unknown_mask = (predictions == 0.5)

        if np.any(unknown_mask):
            predictions[unknown_mask] = self.active_learner.predict(X[unknown_mask])

        return predictions

    def predict_proba(self, X):
        """
        Predicts probabilities using the polytope model first; unknown labels are predicted via the active learner
        """
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
        while data.unknown_size > 0:
            idx_sample, X_sample = data.sample_unlabeled(subsample) if self.rng.random() < self.sample_unknown_proba else data.sample_unknown(subsample)
            idx_selected, X_selected = self.active_learner._select_next(idx_sample, X_sample)

            if self.factorized:
                pred = self.polytope_model.predict_partial(X_selected)
                is_known = (np.min(pred, axis=1) != 0.5)
            else:
                pred = self.polytope_model.predict(X_selected)
                is_known = (pred != 0.5)

            if np.any(is_known):
                data.move_to_labeled([idx for i, idx in enumerate(idx_selected) if is_known[i]], pred[is_known])
                self.__fit_active_learner(data)

            if not np.all(is_known):
                return [idx for i, idx in enumerate(idx_selected) if not is_known[i]], X_selected[~is_known]

    def __fit_active_learner(self, data):
        X, y = data.training_set

        if self.factorized:
            y = np.min(y, axis=1)

        self.active_learner.fit(X, y)
