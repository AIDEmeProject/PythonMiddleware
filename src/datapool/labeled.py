from numpy import array


class LabeledSet:
    """
        In-memory config of all labeled points so far. Classification methods use the labeled set for training.
    """
    def __init__(self):
        self.points = []
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def is_empty(self):
        return len(self.labels) == 0

    def append(self, points, labels):
        self.points.extend(points)
        self.labels.extend(labels)

    def to_array(self):
        return array(self.points), array(self.labels)

    def get_positive_points(self):
        return [pt for pt, lb in zip(self.points, self.labels) if lb == 1]
