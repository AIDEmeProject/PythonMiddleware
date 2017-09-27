from functools import partial
from math import sqrt
from random import uniform

import numpy as np

from .ball import Ball
from .box import Box
from .polytope import Polytope, ActBoostPolytope


def get_line_intersector(K, enclosing=None):
    # get appropriate intersection method
    if isinstance(K, Ball):
        return partial(intersection_ball, ball=K)
    elif isinstance(K, Box):
        return partial(intersection_box, box=K)
    elif isinstance(K, Polytope):
        return partial(intersection_polytope, P=K)
    elif isinstance(K, ActBoostPolytope):
        return partial(intersection_actboostpolytope, P=K)
    elif isinstance(enclosing, Ball):
        return partial(intersection_enclosing, K=K, encloser=intersection_ball)
    elif isinstance(enclosing, Box):
        return partial(intersection_enclosing, K=K, encloser=intersection_box)
    else:
        raise TypeError("Only Ball and Box instances are supported as enclosing body.")


def intersection_enclosing(x, u, K, encloser):
    l1, r2 = encloser.get_intersection(x, u)
    r1 = l2 = 0.0

    while (r2-l1)/(r1-l2) > 1.1:
        mid1 = (l1+r1)/2.0
        if K.is_inside(x + mid1*u):
            r1 = mid1
        else:
            l1 = mid1

        mid2 = (l2 + r2)/2.0
        if K.is_inside(x + mid2*u):
            r2 = mid2
        else:
            l2 = mid2

    while True:
        t = uniform(l1, r2)
        if K.is_inside(x + t*u):
            return t


def intersection_ball(x, u, ball):
    assert isinstance(ball, Ball), "Ball instance expected."

    diff = ball.C - x
    norm = np.sum(u**2)
    b = np.dot(diff, u)
    c = np.sum(np.square(diff)) - ball.R ** 2

    t1 = b - sqrt(b**2 - c*norm)
    t2 = b + sqrt(b**2 - c*norm)

    return uniform(t1/norm, t2/norm)


def intersection_box(x, u, box):
    assert isinstance(box, Box), "Box instance expected."

    lbounds, rbounds = (box.low - x) / u, (box.high - x) / u

    t2 = np.max(np.hstack([lbounds[u > 0], rbounds[u < 0]]))
    t1 = np.min(np.hstack([lbounds[u < 0], rbounds[u > 0]]))

    return uniform(t1, t2)


def intersection_polytope(x, u, P):
    assert isinstance(P, Polytope), "Polytope instance expected."
    assert not P.has_equality or np.allclose(np.dot(P.A, u), 0), "Line x + t*u is not inside P: Au = {}".format(np.dot(P.A, u))

    r1, r2 = [], []
    if P.has_lower:
        lbounds = (P.l - x)/u
        r1.append(lbounds[u > 0])
        r2.append(lbounds[u < 0])
    if P.has_upper:
        rbounds = (P.h - x) / u
        r1.append(rbounds[u < 0])
        r2.append(rbounds[u > 0])
    if P.has_inequality:
        den = np.dot(P.M, u)
        r = (P.q - np.dot(P.M, x))/den
        r1.append(r[den < 0])
        r2.append(r[den > 0])

    return uniform(np.max(np.hstack(r1)), np.min(np.hstack(r2)))


def intersection_actboostpolytope(x, u, P):
    r1, r2 = [], []

    lbounds = -x/u
    r1.append(lbounds[u > 0])
    r2.append(lbounds[u < 0])

    den = np.dot(P.M, u)
    r = -np.dot(P.M, x)/den
    r1.append(r[den < 0])
    r2.append(r[den > 0])

    return uniform(np.max(np.hstack(r1)), np.min(np.hstack(r2)))
