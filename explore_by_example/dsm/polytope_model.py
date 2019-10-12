import numpy as np
from .convex import ConvexHull, ConvexCone

class PolytopeModel:
    def __init__(self):
        self.positive_region = ConvexHull()
        self.negative_regions = []

    def add_labeled_points(self, X, y):
        """
        Increments the polytope model with new labeled data
        Labels must be either 1 (positive) or -1 (negative)
        """
        X, y = np.asmatrix(X), np.asarray(y)

        positive_mask = (y == 1)
        pos_points = X[positive_mask]

        self.positive_region.add_points(pos_points)

        for negative_region in self.negative_regions:
            negative_region.add_points(pos_points)

        for neg_point in X[~positive_mask]:
            self.negative_regions.append(ConvexCone(self.positive_region, neg_point))

    def predict(self, X):
        """
        Predicts the label of each data point.
        1 = positive, -1 = negative, 0 = unknown
        """
        X = np.asmatrix(X)

        predictions = np.zeros(X.shape[0])

        predictions[self.positive_region.is_inside(X)] = 1.0  # all points inside positive region are positive

        # any point inside a negative region is negative
        for negative_region in self.negative_regions:
            predictions[negative_region.is_inside(X)] = -1.0

        return predictions
