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
from libc.math cimport exp, log, log1p, expm1

from cython import boundscheck, wraparound, cdivision
import numpy as np


cdef double __LOGHALF = log(0.5)


@boundscheck(False)
@wraparound(False)
def log1mexp(double[::1] x):
    """
    Computes log(1 - exp(x)) for x < 0
    See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    cdef:
        unsigned int i,  N = x.shape[0]
        double[::1] res = np.empty(N)

    for i in range(N):
        res[i] = log1mexp_single(x[i])

    return res


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def loss(double[::1] x, double[::1] y):
    cdef:
        unsigned int i,  N = x.shape[0]
        double loss = 0

    for i in range(N):
        if y[i] > 0:
            loss -= x[i]
        else:
            loss -= log1mexp_single(x[i])

    loss /= N
    return loss


cdef double log1mexp_single(double x):
    if x < __LOGHALF:
        return log1p(-exp(x))
    else:
        return log(-expm1(x))


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def grad_weights(double[::1] x, double[::1] y):
    cdef:
        unsigned int i, N = x.shape[0]
        double[::1] res = np.empty(N)

    for i in range(N):
        if y[i] > 0:
            res[i] = -1
        else:
            res[i] = 1 / expm1(-x[i])

    return res


@boundscheck(False)
@wraparound(False)
def log_sigmoid(double[::1] x):
    cdef:
        unsigned int i, N = x.shape[0]
        double[::1] res = np.empty(N)

    for i in range(N):
        res[i] = -softmax(-x[i])

    return res

cdef double softmax(double x):
    # Computes softmax(x) = log(1 + exp(x))
    if x >= 0:
        return x + log1p(exp(-x))
    else:
        return log1p(exp(x))
