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
from libc.math cimport sqrt
from cython import boundscheck, wraparound, cdivision


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void partial_cholesky(double[:, ::1] L, double[:, ::1] K):
    cdef int n = K.shape[0], dim = K.shape[1] - 1
    cdef double sum

    for i in range(n):
        sum = 0

        for j in range(dim):
            for k in range(j):
                K[i, j] -= K[i, k] * L[j, k]

            K[i, j] /= L[j, j]
            sum += K[i, j] * K[i, j]

        K[i, dim] = sqrt(1 - sum)  # TODO: replace 1 with kernel(x[j], x[j])
