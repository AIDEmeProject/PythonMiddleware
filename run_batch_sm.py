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
import sys
from time import perf_counter
from typing import Union

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import numpy as np
from aideme.io import read_task
from aideme.utils.random import set_random_state


def train_model_and_measure_time(linear_model, X, y):
    t0 = perf_counter()
    linear_model.fit(X, y)
    return perf_counter() - t0


def read_data(task: str):
    full_data = read_task(task, distinct=True, preprocess=True, read_factorization=False)
    data, labels = full_data['data'], full_data['labels']
    return data.values, labels.values


def build_train_test_random(X, y, train_size: Union[int, float], test_over_all_points: bool = False, stratify: bool = False):
    stratify_array = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=train_size, stratify=stratify_array)
    if test_over_all_points:
        X_test, y_test = X, y
    yield X_train, X_test, y_train, y_test


def build_train_test_kfolds(X, y, k: int, stratify: bool = False):
    kfold_class = StratifiedKFold if stratify else KFold
    kfold = kfold_class(n_splits=k, shuffle=True)
    for train_idx, test_idx in kfold.split(X, y):
        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def print_results(train_scores, test_scores, dts, prefix: str = '', file=None):
    avg_train_score = np.mean(train_scores)
    avg_test_score = np.mean(test_scores)
    avg_dt = np.mean(dts)

    print('{} +- {}\t{} +- {}'.format(avg_train_score, np.std(train_scores) / np.sqrt(len(train_scores)), avg_test_score, np.std(test_scores)/ np.sqrt(len(test_scores))))
    print('{}train = {}, test = {}, fit time = {}'.format(prefix, avg_train_score, avg_test_score, avg_dt), file=file)


# EXPERIMENT CONFIGS
TASK_LIST = [
    'user_study_01', 'user_study_02', 'user_study_03', 'user_study_04', 'user_study_05', 'user_study_06',
    'user_study_07', 'user_study_08', 'user_study_09', 'user_study_10', 'user_study_11', 'user_study_12',
    'user_study_13', 'user_study_14', 'user_study_15', 'user_study_16', 'user_study_17', 'user_study_18',
]
SUBSAMPLE = 0.5
TEST_OVER_ALL = False
STRATIFY = True
K_FOLD = 5  # set to 0 or None to turn off
C = 1e5
SEED = None

# print
print("""TASK_LIST = {}
SUBSAMPLE = {}
TEST_OVER_ALL = {}
STRATIFY = {}
K_FOLD = {}
C = {}
""".format(TASK_LIST, SUBSAMPLE, TEST_OVER_ALL, STRATIFY, K_FOLD, C))

for TASK in TASK_LIST:
    fres = open('./batch_experiments/sm/{}.res'.format(TASK), mode='w')

    try:
        print('------ START TASK {} ------'.format(TASK), file=fres)

        # RUN EXPERIMENT
        set_random_state(SEED)

        # read data
        X, y = read_data(TASK)
        learner = SVC(C=C, gamma=1 / X.shape[1])

        dts = []
        train_scores = []
        test_scores = []

        if K_FOLD is not None and K_FOLD >= 2:
            train_test_generator = build_train_test_kfolds(X, y, k=K_FOLD, stratify=STRATIFY)
        else:
            train_test_generator = build_train_test_random(X, y, train_size=SUBSAMPLE, test_over_all_points=TEST_OVER_ALL, stratify=STRATIFY)

        for X_train, X_test, y_train, y_test in train_test_generator:
            dts.append(train_model_and_measure_time(learner, X_train, y_train))
            train_scores.append(f1_score(y_train, learner.predict(X_train)))
            test_scores.append(f1_score(y_test, learner.predict(X_test)))

        print_results(train_scores, test_scores, dts, prefix='SVM: ', file=fres)

        print('------ END TASK {} ------'.format(TASK), file=fres)
    except Exception:
        print('Task {} failed\n{}'.format(TASK, sys.exc_info()[0]), file=fres)
        fres.close()
