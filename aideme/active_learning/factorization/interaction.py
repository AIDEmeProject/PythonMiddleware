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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def add_interaction_terms(X, self_interacting=False):
    Xs = [X]

    if self_interacting:
        Xs.append(np.square(X))

    D = X.shape[1]
    for i in range(D):
        for j in range(i + 1, D):
            interaction = X[:, i] * X[:, j]
            Xs.append(interaction.reshape(-1, 1))

    return np.hstack(Xs)


def print_interactions(w, self_interacting=False, tol=1e-5):
    D = (np.sqrt(8 * len(w) + 1 + 8 * self_interacting) - 1 - 2 * self_interacting) / 2
    D = int(D)

    w = w[D + D * self_interacting:]
    p = 0
    print('Interaction weights:', w)
    print('Interacting features: ', end='')
    for i in range(D):
        for j in range(i + 1, D):
            if abs(w[p]) > tol:
                print('({}, {}), '.format(i, j), end='')
            p += 1


def print_all_interactions(X, y, Cs=np.linspace(1e-3, 1e-2, 25), self_interacting=False, tol=1e-5):
    X_inter = add_interaction_terms(X, self_interacting)

    for C in Cs:
        print()
        clf = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=10000)
        clf.fit(X_inter, y)

        print('Fscore: ', f1_score(y, clf.predict(X_inter)))
        print_interactions(clf.coef_.ravel(), self_interacting, tol)
        print()
