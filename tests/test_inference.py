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
import numpy as np

from aideme.active_learning.inference.inference import Inference

# TODO: implement real unit tests
class TestInference:
    def test(self):
        X = np.array([
            ['m', 'b', '4'],
            ['l', 'r', '4'],
            ['s', 'y', '2'],
            ['s', 'r', '4'],
            ['m', 'y', '2'],
            ['m', 'y', '4'],
        ])
        y = np.array([1, 1, 0, 0, 0, 1])

        inference = Inference([[0], [1], [2]])
        print()
        for pt, lb in zip(X, y):
            inference._update_single(pt, lb)
            print('------------------')
            print('pt =', pt, ', lb =', lb)
            print(inference)
            print('------------------')
