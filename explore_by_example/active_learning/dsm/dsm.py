import random
import numpy as np
from .convex import ConvexHull, ConvexCone
from ..active_learner import ActiveLearner


class DualSpaceModel(ActiveLearner):
    """
    Dual Space model
    """

    def __init__(self, active_learner, use_al_proba=0.5, seed=None, tol=1e-12):
        self.active_learner = active_learner
        self.use_al_proba = use_al_proba
        self.polytope_model = PolytopeModel(tol)
        self.rng = random.Random(seed)

    def clear(self):
        self.polytope_model = PolytopeModel()

    def fit_data(self, data):
        """
        Fits both active learner and polytope model.
        """
        self.active_learner.fit_data(data)

        X_new, y_new = data.last_labeled_set
        self.polytope_model.add_labeled_points(X_new, y_new)

        unk_idx, unk = data.unknown
        pred = self.polytope_model.predict(unk)
        data.move_to_inferred(unk_idx[pred != -1])

    def predict(self, X):
        """
        Predicts classes based on polytope model first; unknown labels are labeled via the active learner
        """
        predictions = self.polytope_model.predict(X)

        unknown_mask = (predictions == -1)

        if np.any(unknown_mask):
            predictions[unknown_mask] = self.active_learner.predict(X[unknown_mask])

        return predictions

    def predict_proba(self, X):
        """
        Predicts probabilities using the polytope model first; unknown labels are predicted via the active learner
        """
        probas = self.polytope_model.predict_proba(X)

        unknown_mask = (probas == 0.5)
        probas[unknown_mask] = self.active_learner.predict_proba(X[unknown_mask])

        return probas

    def rank(self, X):
        """
        Simply use AL to rank points
        """
        return self.active_learner.rank(X)

    def next_points_to_label(self, data, subsample=None):
        while data.unknown_size > 0:
            idx_sample, X_sample = data.sample_unlabeled(subsample) if self.rng.random() < self.use_al_proba else data.sample_unknown(subsample)
            idx_selected, X_selected = self.active_learner._select_next(idx_sample, X_sample)

            pred = self.polytope_model.predict(X_selected)
            is_known = (pred != -1)

            if np.any(is_known):
                data.move_to_labeled([idx for i, idx in enumerate(idx_selected) if is_known[i]], pred[is_known])
                self.active_learner.fit_data(data)

            if not np.all(is_known):
                return [idx for i, idx in enumerate(idx_selected) if not is_known[i]], X_selected[~is_known]


class PolytopeModel:
    def __init__(self, tol=1e-12):
        self.positive_region = None
        self.negative_regions = []
        self.pos_cache = None
        self.neg_cache = None
        self.tol = tol

    def add_labeled_points(self, X, y):
        """
        Increments the polytope model with new labeled data
        Labels must be either 1 (positive) or 0 (negative)
        """
        X, y = np.asmatrix(X), np.asarray(y)

        # if positive region exists, update positive and negative regions

        positive_mask = (y == 1)
        pos_points = X[positive_mask]

        self.update_positive_region(pos_points)

        for negative_region in self.negative_regions:
            negative_region.add_points(pos_points)

        self.update_negative_region(X[~positive_mask])


    def update_positive_region(self, X):
        """ Updates positive region. If there are not enough points to build this region, we simply cache the points """

        # is positive region exists, update it
        if self.positive_region is not None:
            self.positive_region.add_points(X)
            return

        # update positive points cache
        if self.pos_cache is None:
            self.pos_cache = X.copy()
        else:
            self.pos_cache = np.vstack([self.pos_cache, X])

        # build positive region if enough points have been found
        if len(self.pos_cache) > X.shape[1]:
            self.positive_region = ConvexHull(self.pos_cache, self.tol)
            # self.pos_cache = None

    def update_negative_region(self, X):
        # if negative region exists, update it
        if self.negative_regions or self.positive_region:
            vertices = self.positive_region.vertices if self.positive_region else self.pos_cache
            for neg_point in X:
                self.negative_regions.append(ConvexCone(vertices, neg_point,  self.tol))

            return

        # update negative points cache
        if self.neg_cache is None:
            self.neg_cache = X.copy()
        else:
            self.neg_cache = np.vstack([self.neg_cache, X])

        # if there are at least d positive points, we can start building the negative regions
        if len(self.pos_cache) >= X.shape[1]:
            for neg_point in self.neg_cache:
                self.negative_regions.append(ConvexCone(self.pos_cache, neg_point, self.tol))
                # self.neg_cache = None

    def predict(self, X):
        """
        Predicts the label of each data point.
        1 = positive, -1 = unknown, 0 = negative
        """
        X = np.asmatrix(X)

        predictions = np.full(shape=(X.shape[0],), fill_value=-1.0)  #np.zeros(X.shape[0])

        if self.positive_region:
            predictions[self.positive_region.is_inside(X)] = 1.0  # all points inside positive region are positive

        # any point inside a negative region is negative
        for negative_region in self.negative_regions:
            predictions[negative_region.is_inside(X)] = 0.0

        return predictions

    def predict_proba(self, X):
        """
        Predicts the label probability of each data point.
        1 if positive, 0 if negative, 0.5 otherwise
        """
        X = np.asmatrix(X)

        probas = np.full(shape=(X.shape[0],), fill_value=0.5)

        if self.positive_region:
            probas[self.positive_region.is_inside(X)] = 1.0  # all points inside positive region are certainly positive

        # any point inside a negative region is certainly negative
        for negative_region in self.negative_regions:
            probas[negative_region.is_inside(X)] = 0.0

        return probas
