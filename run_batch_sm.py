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

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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


def build_train_test_sets(X, y, train_size: int, test_over_all_points: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    if test_over_all_points:
        X_test, y_test = X, y
    return X_train, X_test, y_train, y_test


def print_results(learner, X_train, y_train, X_test, y_test, dt, prefix: str = '', file=None):
    train_score = f1_score(y_train, learner.predict(X_train))
    test_score = f1_score(y_test, learner.predict(X_test))
    print('{}train = {}, test = {}, fit time = {}'.format(prefix, train_score, test_score, dt), file=file)


# EXPERIMENT CONFIGS
TASK_LIST = ['sdss_q5', 'sdss_q9']
SUBSAMPLE = 500000
TEST_OVER_ALL = True
C = 1e5
SEED = 10

# print
print("""TASK_LIST = {}
SUBSAMPLE = {}
TEST_OVER_ALL = {}
C = {}
""".format(TASK_LIST, SUBSAMPLE, TEST_OVER_ALL, C))

for TASK in TASK_LIST:
    fres = open('./batch_experiments/sm/{}.res'.format(TASK), mode='w')

    try:
        print('------ START TASK {} ------'.format(TASK), file=fres)

        # RUN EXPERIMENT
        set_random_state(SEED)

        # read data
        X, y = read_data(TASK)
        X_train, X_test, y_train, y_test = build_train_test_sets(X, y, train_size=SUBSAMPLE, test_over_all_points=TEST_OVER_ALL)

        # train SVM
        learner = SVC(C=C, gamma=1 / X.shape[1])
        dt = train_model_and_measure_time(learner, X_train, y_train)
        print_results(learner, X_train, y_train, X_test, y_test, dt, prefix='SVM: ', file=fres)

        print('------ END TASK {} ------'.format(TASK), file=fres)
    except Exception:
        print('Task {} failed\n{}'.format(TASK, sys.exc_info()[0]), file=fres)
        fres.close()
