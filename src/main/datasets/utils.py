import numpy as np

# circle query
def sample_from_unit_sphere(size):
    X = np.random.normal(size=size)
    return X / np.linalg.norm(X, axis=1).reshape(-1, 1)


def sample_from_ball(N, center, radius=1.0):
    center = np.array(center).ravel()
    dim = len(center)

    X = sample_from_unit_sphere(size=(N, dim))
    X *= np.power(np.random.uniform(low=0, high=1, size=(N, 1)), 1.0 / dim)

    return radius * X + center


def sample_from_annulus(N, center, inner=1.0, outer=2.0):
    if outer <= inner:
        raise ValueError("inner >= outer")

    center = np.array(center).ravel()
    dim = len(center)

    X = sample_from_unit_sphere(size=(N, dim))
    U = np.random.uniform(low=0, high=1, size=(N, 1))
    factor = inner * (1 + U * ((outer / inner) ** dim - 1)) ** (1.0 / dim)

    return center + factor * X
