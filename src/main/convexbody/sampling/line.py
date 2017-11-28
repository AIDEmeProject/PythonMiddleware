import numpy as np


class Line:
    def __init__(self, center, direction):
        self.center = np.array(center, dtype=np.float64).ravel()
        self.direction = np.array(direction, dtype=np.float64).ravel()

        if len(self.center) != len(self.direction):
            raise ValueError("Point and direction have different dimensions!")

        self.direction = self.direction / np.linalg.norm(self.direction)

    def __repr__(self):
        return "Center: {0}, Direction: {1}".format(self.center, self.direction)

    @classmethod
    def sample_line(cls, point, projection_matrix=None):
        direction = np.random.normal(size=len(point))

        if projection_matrix is not None:
            direction = projection_matrix.dot(direction)

        return cls(point, direction)

    def get_segment(self, left_limit, right_limit):
        return LineSegment(self, left_limit, right_limit)


class LineSegment:
    def __init__(self, line, left_limit, right_limit):
        self.line = line
        self.left_limit = float(left_limit)
        self.right_limit = float(right_limit)

        if self.left_limit >= self.right_limit:
            raise ValueError("Left limit must be smaller than right!")

    def sample(self, n_samples):
        values = np.random.uniform(low=self.left_limit, high=self.right_limit, size=(n_samples, 1))
        return self.line.center + values * self.line.direction

    def get_extremes(self):
        return (
            self.line.center + self.left_limit * self.line.direction,
            self.line.center + self.right_limit * self.line.direction
        )
