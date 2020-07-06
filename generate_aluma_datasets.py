#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from time import perf_counter
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from aideme.active_learning.aluma import aluma_preprocessing
from aideme.active_learning.kernel import IncrementedDiagonalKernel, GaussianKernel
from aideme.utils import assert_positive_integer, assert_in_range
from aideme.utils.random import get_random_state


def generate_data(size: int, dim: int, selectivity: float, seed: Optional[Union[int, np.random.RandomState]] = None):
    assert_positive_integer(size, 'size')
    assert_positive_integer(dim, 'dim')
    assert_in_range(selectivity, 'selectivity', 0, 1)

    rng = get_random_state(seed)

    length = np.sqrt(3)
    limit = (2 * length / np.sqrt(np.pi)) * (selectivity * gamma(1 + dim / 2)) ** (1 / dim)

    X = rng.uniform(-length, length, (size, dim))
    y = (np.linalg.norm(X, axis=1) < limit).astype('float')

    if limit > np.sqrt(3):
        print('Exceeded bounds. Selectivity =', y.sum() / len(y))

    return X, y

def assert_linear_separable(X: np.ndarray, y: np.ndarray) -> None:
    clf = SVC(C=1e5, kernel='linear')
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert f1_score(y, y_pred) == 1, "Data is not linearly separable"

# Global params
SEED = 0
KERNEL = IncrementedDiagonalKernel(GaussianKernel(), jitter=1e-12)
ASSERT_SEPARABLE = True

# Dataset params
SIZE = int(1e4)
DIM = 2
SELECTIVITY = 0.01

# Aluma params
MARGIN = 1.3
DELTA = 1.0
H = 0

#######################
# BEGIN ALGORITHM
#######################
print("""--------ALuMa Dataset Generator---------
Dataset: size = {}, dim = {}, selec = {}
ALuMa: margin = {}, delta = {}, H = {}
""".format(SIZE, DIM, SELECTIVITY, MARGIN, DELTA, H))

rng = np.random.RandomState(SEED)

# Data generation
X, y = generate_data(SIZE, DIM, SELECTIVITY, rng)
print('Generated data. # positive points =', y.sum())

# ALuMa preprocessing
t0 = perf_counter()
X_aluma = aluma_preprocessing(X, KERNEL, MARGIN, H, DELTA, rng)
print("Finished ALuMa preprocessing in {} seconds. Output dim = {}".format(perf_counter() - t0, X_aluma.shape[1]))

if ASSERT_SEPARABLE:
    assert_linear_separable(X_aluma, y)
    print('Data is linearly separable.')

# Saving results to disk
df = pd.DataFrame(X)
df['labels'] = y
df.to_csv('./data/original_size={}_dim={}.tsv'.format(SIZE, DIM), sep='\t', index=False)

df = pd.DataFrame(X_aluma)
df['labels'] = y
df.to_csv('./data/aluma_size={}_dim={}.tsv'.format(SIZE, DIM), sep='\t', index=False)

print("----------------------------------------")
