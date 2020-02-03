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

import math


def assert_positive(value, name):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("{0} must be a positive number, got {1}".format(name, value))


def assert_positive_integer(value, name, allow_inf=False, allow_none=False):
    assert_non_negative_integer(value, name, allow_inf, allow_none)
    if value == 0:
        raise ValueError("Expected positive integer for {}, got 0".format(name))

def assert_non_negative_integer(value, name, allow_inf=False, allow_none=False):
    if value is None:
        if not allow_none:
            raise ValueError("{} cannot be none.".format(name))
        return

    if value == math.inf:
        if not allow_inf:
            raise ValueError("{0} cannot be infinity.".format(name))
        return

    if not isinstance(value, int) or value < 0:
        raise ValueError("{0} must be a positive integer, got {1}".format(name, value))

def process_callback(callback):
    if not callback:
        return []

    if callable(callback):
        return [callback]

    if not all(callable(f) for f in callback):
        raise ValueError("Expected callable or list of callable objects, got {}".format(callback))

    return callback
