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
from libc.math cimport sqrt, INFINITY
from libc.stdlib cimport malloc, free

from cython import boundscheck, wraparound, nonecheck, cdivision

from scipy.linalg.cython_blas cimport dgemv


cpdef get_polytope_extremes(double[::1] num, double[::1] den):
    cdef int n = num.shape[0]
    cdef double* res = compute_extremes_min_max(&num[0], &den[0], n)
    return res[0], res[1]


cpdef get_polytope_extremes_opt(double[:, ::1] A, double[::1] center, double[::1] direction):
    cdef int m = A.shape[1], n = A.shape[0]
    cdef double* res = compute_pol_extremes(&A[0, 0], &center[0], &direction[0], m, n)
    return res[0], res[1]


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(False)
cdef double* compute_pol_extremes(double* A, double* center, double* direction, int m, int n) nogil:
    cdef:
        char *transa = 't'
        int inc = 1, i
        double alpha = 1, beta = 0, e
        double *num = <double*> malloc(sizeof(double) * m)
        double *den = <double*> malloc(sizeof(double) * m)

    dgemv(transa, &m, &n, &alpha,  A, &m, center, &inc, &beta, num, &inc)
    dgemv(transa, &m, &n, &alpha, A, &m, direction, &inc, &beta, den, &inc)

    cdef double* res = compute_extremes_min_max(num, den, n)

    free(num)
    free(den)

    return res


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(False)
cdef double* compute_extremes_min_max(double* num, double* den, unsigned int n) nogil:
    cdef:
        unsigned int i
        double l = -INFINITY, u = INFINITY, d, e

    for i in range(n):
        d = den[i]

        if d != 0:
            e = - num[i] / d

            if d < 0:
                if e > l:
                    l = e

            elif e < u:
                 u = e

    cdef double *vals = <double*> malloc(sizeof(double) * 2)
    vals[0] = l
    vals[1] = u
    return vals


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(False)
cpdef get_ball_extremes(double[::1] center, double[::1] direction):
    cdef:
        unsigned int i
        double a = 0, b = 0, c = -1
        double delta, sq_delta

    for i in range(len(center)):
        a += direction[i] * direction[i]
        b += center[i] * direction[i]
        c += center[i] * center[i]

    delta = b ** 2 - a * c

    if delta <= 0:
        raise RuntimeError("Line does not intersect unit ball.")

    sq_delta = sqrt(delta)
    return (-b - sq_delta) / a, (-b + sq_delta) / a
