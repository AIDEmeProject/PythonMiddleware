import numpy as np
from pandas import DataFrame, Series

from src.main.user import DummyUser
from .utils import sample_from_ball

def xor_query(N, sel):
    if N <= 4:
        raise ValueError()
    if sel <= 0 or sel >= 1:
        raise ValueError()

    N_pos = int(sel*N)
    N_neg = N - N_pos

    samples = []

    # positive points
    center = np.ones(2)
    samples.append(sample_from_ball(N_pos//2, center, 1))
    samples.append(sample_from_ball(N_pos - N_pos//2, -center, 1))

    # negative points
    center[-1] = -1
    samples.append(sample_from_ball(N_neg//2, center, 1))
    samples.append(sample_from_ball(N_neg - N_neg//2, -center, 1))

    labels = [1]*N_pos + [-1]*N_neg

    # data and user
    data = DataFrame(np.vstack(samples))
    user = DummyUser(Series(labels), max_iter=100)

    return data, user