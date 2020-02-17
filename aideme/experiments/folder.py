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

import datetime
import json
import os
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..utils import Config


class ExperimentFolder:
    def __init__(self, folder: str):
        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)

    @property
    def path(self) -> str:
        return self._folder

    def read_config(self) -> Config:
        path = os.path.join(self._folder, 'config.json')

        with open(path, 'r') as f:
            return json.load(f)

    def write_config(self, config: Config) -> None:
        path = os.path.join(self._folder, 'config.json')

        with open(path, 'w') as f:
            return json.dump(config, f, indent=4)

    def save(self, df: pd.DataFrame, filename: str) -> None:
        path = os.path.join(self._folder, filename)
        df.to_csv(path, sep='\t', index_label='iter')

    def save_run(self, run, filename):
        path = os.path.join(self._folder, filename)

        run_metrics = {}
        with open(path, 'w') as file:
            for i, metrics in enumerate(run):
                json.dump(metrics, file, cls=JsonEncoder)
                file.write('\n')
                run_metrics[i] = metrics

        return pd.DataFrame.from_dict(run_metrics, orient='index')

    def read_run_files(self):
        return [pd.read_csv(os.path.join(self._folder, file), sep='\t', index_col='iter') for file in os.listdir(self._folder) if file.startswith('run')]

    def delete(self, filename) -> None:
        path = os.path.join(self._folder, filename)
        os.remove(path)


class RootFolder:
    def __init__(self):
        now = datetime.datetime.now()
        self._root = os.path.join('experiments', str(now))

    @property
    def log_file(self) -> str:
        return os.path.join(self._root, 'experiment.log')

    def get_experiment_folder(self, task: str, learner: str) -> ExperimentFolder:
        folder = os.path.join(self._root, task, learner)
        return ExperimentFolder(folder)

    def get_all_tasks(self) -> List[str]:
        return self.__get_all_files(self._root)

    def get_all_learners(self, task: str) -> List[str]:
        task_folder = os.path.join(self._root, task)
        return self.__get_all_files(task_folder)

    @staticmethod
    def __get_all_files(path):
        return [f for f in os.listdir(path) if not (f.endswith('.log') or f.startswith('.'))]


class JsonEncoder(json.JSONEncoder):
    """
    Custom JsonEncoder supporting numpy data types
    """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
