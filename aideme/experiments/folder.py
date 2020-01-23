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


if TYPE_CHECKING:
    import pandas as pd
    from ..utils import Config


class ExperimentFolder:
    def __init__(self, folder: str):
        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)

    @property
    def folder(self) -> str:
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


class RootFolder:
    def __init__(self):
        now = datetime.datetime.now()
        self._root = os.path.join('experiments', str(now))

    def get_experiment_folder(self, task: str, learner: str) -> ExperimentFolder:
        folder = os.path.join(self._root, task, learner)
        return ExperimentFolder(folder)

    def get_all_tasks(self) -> List[str]:
        return os.listdir(self._root)

    def get_all_learners(self, task: str) -> List[str]:
        task_folder = os.path.join(self._root, task)
        return os.listdir(task_folder)
