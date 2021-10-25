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
