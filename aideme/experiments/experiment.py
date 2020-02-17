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

from typing import TYPE_CHECKING

from .decoder import read_training_set, decode_active_learner, build_exploration_object
from .folder import RootFolder
from .logger import ExperimentLogger

if TYPE_CHECKING:
    from ..utils import Config, RunsType


def run_all_experiments(root_folder: RootFolder) -> None:
    logger = ExperimentLogger(root_folder)

    for task in root_folder.get_all_tasks():
        training_set = read_training_set(task)

        for learner in root_folder.get_all_learners(task):
            exp_folder = root_folder.get_experiment_folder(task, learner)

            try:
                logger.experiment_begin(task, learner)
                runs = run_experiment(exp_folder.read_config(), training_set)

                for i, run in enumerate(runs):
                    filename = 'run_{0:0=2d}'.format(i + 1)

                    # run experiment and save metrics as they are computed
                    logger.run_begin()
                    df = exp_folder.save_run(run, filename + '.tmp')

                    # remove tmp file and save csv
                    exp_folder.save(df, filename + '.tsv')
                    exp_folder.delete(filename + '.tmp')

            except Exception as e:
                logger.error(task, learner, exception=e)
                continue  # Move to next experiment

    logger.end()

    if logger.errors == 0:
        compute_average(root_folder)


def run_experiment(config: Config, training_set=None) -> RunsType:
    if training_set is not None:
        X, labeled_set, factorization_info = training_set
    else:
        X, labeled_set, factorization_info = read_training_set(config['task'])

    # build exploration object and active learner
    active_learner = decode_active_learner(config['active_learner'], factorization_info)
    exploration = build_exploration_object(config, labeled_set)

    # run experiment
    return exploration.run(X, labeled_set, active_learner, repeat=config['repeat'], seeds=config['seeds'], return_generator=True)


def compute_average(root_folder: RootFolder) -> None:
    for task in root_folder.get_all_tasks():
        for learner in root_folder.get_all_learners(task):
            exp_folder = root_folder.get_experiment_folder(task, learner)

            runs = exp_folder.read_run_files()
            to_keep = [col for col, tp in zip(runs[0].columns, runs[0].dtypes) if tp in ('int', 'float')]
            avg = sum((df[to_keep] for df in runs)) / len(runs)

            exp_folder.save(avg, 'average.tsv')
