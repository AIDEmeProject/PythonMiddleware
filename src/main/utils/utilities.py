import numpy as np

def label_all(data, user):
    return user.get_label(data, update_counter=False)

def check_labels(labels):
    labels = np.array(labels, dtype=np.float64).ravel()
    if not set(labels) <= {-1, 1}:
        raise ValueError("Labels must be either -1 or 1")

    return labels

def check_points(points):
    points = np.array(points, dtype=np.float64)
    if len(points.shape) == 1:
        return np.atleast_2d(points)
    elif len(points.shape) == 2:
        return points
    raise ValueError("Received object with more than 2 dimensions!")

def check_points_and_labels(points, labels):
    points = check_points(points)
    labels = check_labels(labels)

    check_sizes(points, labels)

    return points, labels

def check_sizes(a, b):
    if len(a) != len(b):
        raise ValueError("Incompatible dimensions")
