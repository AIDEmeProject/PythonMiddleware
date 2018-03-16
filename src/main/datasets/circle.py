import numpy as np
import pandas as pd
from src.main.user import DummyUser

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


def circle_query(N, center, sel, sep):
    N_in = int(N * sel)
    N_out = N - N_in

    center = np.array(center).ravel()
    dim = len(center)
    r_sep = sel ** (1.0 / dim)

    X_out = sample_from_annulus(N_out, center, inner=sep + (1 - sep) * r_sep, outer=1.0)
    X_in = sample_from_ball(N_in, center, r_sep)
    X = np.vstack([X_out, X_in])

    y = np.array([-1] * N_out + [1] * N_in)

    data = pd.DataFrame(X)
    user = DummyUser(pd.Series(y), max_iter=50)
    return data, user
