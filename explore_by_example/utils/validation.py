import math


def assert_positive_integer(value, name, allow_inf=False):
    if value == math.inf and not allow_inf:
        raise ValueError("{0} cannot be infinity.".format(name))

    if not isinstance(value, int) or value <= 0:
        raise ValueError("{0} must be a positive integer, got {1}".format(name, value))

