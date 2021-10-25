#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.

import math

def assert_in_range(value, name, low, high):
    if value < low or value > high:
        raise ValueError("Expected {} < {} < {}, but got {}".format(low, name, high, value))

def assert_positive(value, name, allow_inf=False, allow_none=False):
    __assert_positive((int, float), value, name, allow_inf, allow_none)

def assert_non_negative(value, name, allow_inf=False, allow_none=False):
    __assert_non_negative((int, float), value, name, allow_inf, allow_none)

def assert_positive_integer(value, name, allow_inf=False, allow_none=False):
    __assert_positive(int, value, name, allow_inf, allow_none)

def assert_non_negative_integer(value, name, allow_inf=False, allow_none=False):
    __assert_non_negative(int, value, name, allow_inf, allow_none)

def __assert_positive(type, value, name, allow_inf=False, allow_none=False):
    __assert_non_negative(type, value, name, allow_inf, allow_none)
    if value == 0:
        raise ValueError("Expected positive '{}', got 0".format(name))

def __assert_non_negative(type, value, name, allow_inf=False, allow_none=False):
    if value is None:
        if not allow_none:
            raise ValueError("{} cannot be none.".format(name))
        return

    if value == math.inf:
        if not allow_inf:
            raise ValueError("{} cannot be infinity.".format(name))
        return

    if not isinstance(value, type) or value < 0:
        raise ValueError("'{}' must be a positive, got {}".format(name, value))

def process_callback(callback):
    if not callback:
        return []

    if callable(callback):
        return [callback]

    if not all(callable(f) for f in callback):
        raise ValueError("Expected callable or list of callable objects, got {}".format(callback))

    return callback
