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

from cython import boundscheck, wraparound, cdivision, NULL
from scipy.linalg.cython_blas cimport dgemv


@boundscheck(False)
@wraparound(False)
def compute_version_space_intersection(double[:, ::1] A, double[::1] center, double[::1] direction):
    cdef int m = A.shape[0], n = A.shape[1]
    cdef double* res = compute_version_space_intersection_opt(&A[0, 0], &center[0], &direction[0], m, n)

    if res == NULL:
        raise RuntimeError("Line does not intersect version space.")

    return res[0], res[1]


@boundscheck(False)
@wraparound(False)
cdef double* compute_version_space_intersection_opt(double* A, double* center, double* direction, int m, int n):
    cdef:
        double *extremes, *ball_extremes
        double lower, upper

    extremes = compute_polytope_intersection(A, center, direction, m, n)
    if extremes == NULL:
        return NULL

    ball_extremes = compute_ball_intersection(center, direction, n)
    if ball_extremes == NULL:
        return NULL

    if ball_extremes[0] > extremes[0]:
        extremes[0] = ball_extremes[0]

    if ball_extremes[1] < extremes[1]:
        extremes[1] = ball_extremes[1]

    free(ball_extremes)

    return extremes


cdef double* compute_polytope_intersection(double* A, double* center, double* direction, int m, int n):
    cdef:
        char *transa = 't'
        int inc = 1
        double alpha = 1, beta = 0
        double *num = <double*> malloc(sizeof(double) * n)
        double *den = <double*> malloc(sizeof(double) * n)

    dgemv(transa, &n, &m, &alpha,  A, &n, center, &inc, &beta, num, &inc)
    dgemv(transa, &n, &m, &alpha, A, &n, direction, &inc, &beta, den, &inc)

    cdef double* vals = polytope_intersection_min_max(num, den, m)
    free(num)
    free(den)
    return vals


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double* polytope_intersection_min_max(double* num, double* den, unsigned int m):
    cdef:
        unsigned int i
        double l = -INFINITY, u = INFINITY, d, n, e

    for i in range(m):
        n = num[i]
        d = den[i]

        if d < 0:
            e =  - n / d
            if e > l:
                l = e
        elif d > 0:
            e =  - n / d
            if e < u:
                u = e
        elif n > 0:
            return NULL

    if l >= u:
        return NULL

    return create_two_dimensional_array(l, u)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double* compute_ball_intersection(double* center, double* direction, unsigned int n):
    cdef:
        unsigned int i
        double a = 0, b = 0, c = -1
        double di, ci, delta, sq_delta

    for i in range(n):
        ci = center[i]
        di = direction[i]
        a += di * di
        b += ci * di
        c +=ci * ci

    delta = b ** 2 - a * c
    if delta <= 0:
        return NULL
    sq_delta = sqrt(delta)

    return create_two_dimensional_array((-b - sq_delta) / a, (-b + sq_delta) / a)


@boundscheck(False)
@wraparound(False)
cdef double* create_two_dimensional_array(double a, double b):
    cdef double *vals = <double*> malloc(sizeof(double) * 2)
    vals[0] = a
    vals[1] = b
    return vals
