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
import os
from time import perf_counter
from typing import Optional, Union

import numpy as np

from aideme.active_learning.aluma import aluma_preprocessing
from aideme.active_learning.kernel import IncrementedDiagonalKernel, GaussianKernel
from aideme.utils import assert_positive_integer, assert_in_range
from aideme.io import read_task
from aideme.io.utils import get_config_from_resources, write_config_to_resources
from aideme.utils.random import get_random_state


def get_path_to_data_folder():
    return get_config_from_resources('sources', 'filesystem')['path']


def update_config(filename):
    config_name = filename.rsplit('.', 1)[0]

    write_config_to_resources(
        'datasets',
        config_name,
        {
            'source': 'filesystem',
            'filename': filename,
        }
    )

    write_config_to_resources(
        'tasks',
        config_name,
        {
            'dataset': {
                'tag': config_name
            }
        }
    )


def save_data(X, y, path):
    import pandas as pd

    df = pd.DataFrame(X)
    df['labels'] = y
    df.to_csv(path, sep='\t', index=False)


def generate_data(size: int, dim: int, selectivity: float, seed: Optional[Union[int, np.random.RandomState]] = None):
    from scipy.special import gamma

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


def read_cars_query(query_number):
    data = read_task('user_study_' + query_number)
    return data['data'].values, data['labels'].values


def assert_linear_separable(X: np.ndarray, y: np.ndarray) -> None:
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

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

CAR_QUERY = 2
IS_CARS = CAR_QUERY is not None
CAR_TASK = str(CAR_QUERY) if CAR_QUERY >= 10 else '0' + str(CAR_QUERY) if IS_CARS else None

# Aluma params
MARGIN = 1.3
DELTA = 1.0
H = 0

#######################
# BEGIN ALGORITHM
#######################

dataset_desc = 'cars {}'.format(CAR_TASK) if IS_CARS else 'size = {}, dim = {}, selec = {}'.format(SIZE, DIM, SELECTIVITY)
print("""--------ALuMa Dataset Generator---------
Dataset: {}
ALuMa: margin = {}, delta = {}, H = {}
""".format(dataset_desc, MARGIN, DELTA, H))

rng = get_random_state(SEED)

# Data generation
X, y = read_cars_query(CAR_TASK) if IS_CARS else generate_data(SIZE, DIM, SELECTIVITY, rng)
print('Generated data. # positive points =', y.sum())

# ALuMa preprocessing
t0 = perf_counter()
X_aluma = aluma_preprocessing(X, KERNEL, MARGIN, H, DELTA, rng)
print("Finished ALuMa preprocessing in {} seconds. Output dim = {}".format(perf_counter() - t0, X_aluma.shape[1]))

if ASSERT_SEPARABLE:
    assert_linear_separable(X_aluma, y)
    print('Data is linearly separable.')

# Saving results to disk
path_to_data_folder = get_path_to_data_folder()

if not IS_CARS:
    filename = 'aluma_size={}_dim={}_original.tsv'.format(SIZE, DIM)
    path = os.path.join(path_to_data_folder, filename)
    save_data(X, y, path)
    update_config(filename)

filename = 'aluma_user_study_{}_preprocessed.tsv'.format(CAR_TASK) if IS_CARS else 'aluma_size={}_dim={}_preprocessed.tsv'.format(SIZE, DIM)
path = os.path.join(path_to_data_folder, filename)
save_data(X_aluma, y, path)
update_config(filename)

print("----------------------------------------")
