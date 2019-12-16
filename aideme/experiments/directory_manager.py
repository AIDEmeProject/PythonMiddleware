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
from datetime import datetime

from pandas import read_csv

from definitions import ROOT_DIR


class ExperimentDirManager:
    def __init__(self):
        self.root = os.path.join(ROOT_DIR, 'experiments')
        self.experiment_folder = None

    def set_new_experiment_folder(self):
        now = datetime.now()
        self.experiment_folder = os.path.join(self.root, 'tmp', str(now))
        os.makedirs(self.experiment_folder)

    def get_data_folder(self, data_tag, learner_tag):
        return DataFolder(self.experiment_folder, data_tag, learner_tag)


class DataFolder:
    def __init__(self, experiment_folder, data_tag, learner_tag):
        self.path = os.path.join(experiment_folder, data_tag, learner_tag)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_full_path(self, file):
        return os.path.join(self.path, file)

    def get_files(self, filter_str=''):
        return map(self.get_full_path,  filter(lambda name: filter_str in name, os.listdir(self.path)))

    def get_raw_runs(self):
        raw_files = self.get_files('raw')
        return (read_csv(file, sep='\t', index_col='iter') for file in raw_files)

    def read_average(self):
        return read_csv(self.get_full_path('average_fscore.tsv'), sep='\t')

    def compute_runs_average(self):
        avg, count = 0., 0

        for run in self.get_raw_runs():
            to_keep = [col for col, tp in zip(run.columns, run.dtypes) if tp != 'object']
            avg += run[to_keep]
            count += 1

        if count > 0:
            avg /= count
            self.write(avg, 'average.tsv')

    def write(self, data, filename):
        path = os.path.join(self.path, filename)
        data.to_csv(path, sep='\t', header=True, index=True, index_label='iter', float_format='%.6f')
