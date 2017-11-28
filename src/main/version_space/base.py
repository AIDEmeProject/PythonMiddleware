from ..utils import check_points_and_labels


class VersionSpace:
    def __init__(self):
        pass

    def clear(self):
        pass

    def update(self, points, labels):
        points, labels = check_points_and_labels(points, labels)
        for point, label in zip(points, labels):
            self._update_single(point, label)

    def _update_single(self, point, label):
        pass

    def sample(self, n):
        raise NotImplementedError
