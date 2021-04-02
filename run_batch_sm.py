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
from typing import Union

import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_validate
from sklearn.svm import SVC

from aideme.io import read_task
from aideme.utils.random import set_random_state


def read_data(task: str):
    full_data = read_task(task, distinct=True, preprocess=True, read_factorization=False)
    data, labels = full_data['data'], full_data['labels']
    return data.values, labels.values


def build_train_test_shuffle(train_size: Union[int, float], repeat: int = 1, stratify: bool = False):
    shuffle_class = StratifiedShuffleSplit if stratify else ShuffleSplit
    return shuffle_class(n_splits=repeat, train_size=train_size)


def build_train_test_kfolds(k: int, repeats: int = 1, stratify: bool = False):
    kfold_class = RepeatedStratifiedKFold if stratify else RepeatedKFold
    return kfold_class(n_splits=k, n_repeats=repeats)


def print_results(train_scores, test_scores, dts, prefix: str = '', file=None):
    avg_train_score = np.mean(train_scores)
    avg_test_score = np.mean(test_scores)
    avg_dt = np.mean(dts)

    factor = np.sqrt(len(train_scores))
    print('{:.2f} +- {:.2f}\t{:.2f} +- {:.2f}'.format(avg_train_score, np.std(train_scores) / factor, avg_test_score, np.std(test_scores) / factor))
    print('{}train = {}, test = {}, fit time = {}'.format(prefix, avg_train_score, avg_test_score, avg_dt), file=file)


# EXPERIMENT CONFIGS
TASK_LIST = [
    #'sdss_q5', 'sdss_q6', 'sdss_q7', 'sdss_q8', 'sdss_q9', 'sdss_q10', 'sdss_q11',
    'user_study_01', 'user_study_02', 'user_study_03', 'user_study_04', 'user_study_05', 'user_study_06',
    'user_study_07', 'user_study_08', 'user_study_09', 'user_study_10', 'user_study_11', 'user_study_12',
    'user_study_13', 'user_study_14', 'user_study_15', 'user_study_16', 'user_study_17', 'user_study_18',
]
TRAIN_SIZE = 0.7
STRATIFY = True
K_FOLD = 5  # set to 0 or None to turn off
REPEATS = 100
C = 1e3
SEED = None

learner = SVC(C=C, gamma='auto')

# print
print("""TASK_LIST = {}
TRAIN_SIZE = {}
STRATIFY = {}
REPEAT = {}
K_FOLD = {}
C = {}
""".format(TASK_LIST, TRAIN_SIZE, STRATIFY, REPEATS, K_FOLD, C))

for TASK in TASK_LIST:
    fres = open('./batch_experiments/sm/{}.res'.format(TASK), mode='w')

    try:
        print('------ START TASK {} ------'.format(TASK), file=fres)

        # RUN EXPERIMENT
        set_random_state(SEED)

        # read data
        X, y = read_data(TASK)

        if K_FOLD is not None and K_FOLD >= 2:
            cv = build_train_test_kfolds(k=K_FOLD, repeats=REPEATS, stratify=STRATIFY)
        else:
            cv = build_train_test_shuffle(train_size=TRAIN_SIZE, repeat=REPEATS, stratify=STRATIFY)
        scores = cross_validate(learner, X, y, scoring='f1', cv=cv, return_train_score=True)

        train_scores = scores['train_score']
        test_scores = scores['test_score']
        dts = scores['fit_time']
        print_results(train_scores, test_scores, dts, prefix='SVM: ', file=fres)

        print('------ END TASK {} ------'.format(TASK), file=fres)
    except Exception:
        print('Task {} failed\n{}'.format(TASK, sys.exc_info()[0]), file=fres)
        fres.close()
