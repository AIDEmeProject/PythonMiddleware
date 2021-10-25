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

from aideme.active_learning.inference.inference import Inference

# TODO: implement real unit tests
class TestInference:
    def test(self):
        X = np.array([
            ['m', 'b', '4'],
            ['l', 'r', '4'],
            ['s', 'y', '2'],
            ['s', 'r', '4'],
            ['m', 'y', '2'],
            ['m', 'y', '4'],
        ])
        y = np.array([1, 1, 0, 0, 0, 1])

        inference = Inference([[0], [1], [2]])
        print()
        for pt, lb in zip(X, y):
            inference._update_single(pt, lb)
            print('------------------')
            print('pt =', pt, ', lb =', lb)
            print(inference)
            print('------------------')
