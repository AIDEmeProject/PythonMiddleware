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
from libc.math cimport exp, log, log1p, expm1

from cython import boundscheck, wraparound, cdivision
import numpy as np


cdef double __LOGHALF = log(0.5)


@boundscheck(False)
@wraparound(False)
cpdef double[::1] compute_log_probas(double[:, ::1] margin):
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = margin.shape[0], K = margin.shape[1]
        double[::1] log_probas = np.empty(N, dtype=np.float64)

    for i in range(N):
        log_probas[i] = 0
        for j in range(K):
            log_probas[i] -= softmax(-margin[i, j])

    return log_probas


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def compute_loss(double[:, ::1] margin, double[::1] y):
    cdef:
        Py_ssize_t i, N = margin.shape[0]
        double lp, loss = 0
        double[::1] log_probas = compute_log_probas(margin)

    for i in range(N):
        lp = log_probas[i]

        if y[i] > 0:
            loss -= y[i] * lp
        else:
            loss -= log1mexp(lp)

    loss /= N
    return loss


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def compute_huber_penalty(double[:, :] weights, double penalty, double delta):
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = weights.shape[0], K = weights.shape[1]
        double loss = 0, half_delta = delta / 2, w

    for i in range(N):
        for j in range(K):
            w = weights[i, j]

            if w > delta:
                loss += delta * (w - half_delta)

            elif w < -delta:
                loss -= delta * (w + half_delta)

            else:
                loss += 0.5 * w * w

    loss *= penalty
    return loss


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def compute_classification_loss(double[::1] log_probas, double[::1] y):
    cdef:
        unsigned int i,  N = log_probas.shape[0]
        double loss = 0

    for i in range(N):
        if y[i] > 0:
            loss -= y[i] * log_probas[i]
        else:
            loss -= log1mexp(log_probas[i])

    loss /= N
    return loss


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def grad_weights(double[::1] x, double[::1] y):
    cdef:
        unsigned int i, N = x.shape[0]
        double[::1] res = np.empty(N)

    for i in range(N):
        res[i] = -y[i] if y[i] > 0 else 1. / expm1(-x[i])

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


cdef double log1mexp(double x):
    """
    Computes log(1 - exp(x)) for x < 0
    See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x < __LOGHALF:
        return log1p(-exp(x))
    else:
        return log(-expm1(x))


cdef double softmax(double x):
    # Computes softmax(x) = log(1 + exp(x))
    if x >= 0:
        return x + log1p(exp(-x))
    else:
        return log1p(exp(x))


@cdivision(True)
cdef double msigmoid(double x):
    # Computes sigmoid(x) = 1 / (1 + exp(x))
    cdef double expx
    if x <= 0:
        expx = exp(x)
        return 1 / (1 + expx)
    else:
        expx = exp(-x)
        return expx / (1 + expx)

