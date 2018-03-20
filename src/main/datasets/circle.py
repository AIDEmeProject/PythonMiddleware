from pandas import DataFrame, Series
from src.main.user import DummyUser
from .utils import *

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

    data = DataFrame(X)
    user = DummyUser(Series(y), max_iter=100)
    return data, user

