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
from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from .decoder import read_training_set, decode_active_learner, build_exploration_object
from .folder import RootFolder
from .logger import ExperimentLogger

if TYPE_CHECKING:
    from ..utils import Config


# TODO: Save metrics to file as soon as they are computed (add logger?)
def run_all_experiments(root_folder: RootFolder) -> None:
    logger = ExperimentLogger(root_folder)

    for task in root_folder.get_all_tasks():
        training_set = read_training_set(task)

        for learner in root_folder.get_all_learners(task):
            exp_folder = root_folder.get_experiment_folder(task, learner)

            try:
                logger.start(task, learner)
                runs = run_experiment(exp_folder.read_config(), training_set)
                logger.finish(task, learner)
            except Exception as e:
                logger.error(task, learner, exception=e)
                continue  # Move to next experiment

            # save metrics to disk
            for i, df in enumerate(runs):
                str_i = str(i + 1) if i >= 9 else '0' + str(i + 1)
                exp_folder.save(df, 'run_{}.tsv'.format(str_i))

            # compute average and save to disk
            to_keep = [col for col, tp in zip(runs[0].columns, runs[0].dtypes) if tp in ('int', 'float')]
            avg = sum((run[to_keep] for run in runs)) / len(runs)
            exp_folder.save(avg, 'average.tsv')

    logger.end()


def run_experiment(config: Config, training_set=None) -> List[pd.DataFrame]:
    if training_set is not None:
        X, labeled_set, factorization_info = training_set
    else:
        X, labeled_set, factorization_info = read_training_set(config['task'])

    # build exploration object and active learner
    active_learner = decode_active_learner(config['active_learner'], factorization_info)
    exploration = build_exploration_object(config, labeled_set)

    # run experiment
    runs = exploration.run(X, labeled_set, active_learner, repeat=config['repeat'], seeds=config['seeds'])
    return [pd.DataFrame.from_dict({i: metrics for i, metrics in enumerate(run)}, orient='index') for run in runs]
