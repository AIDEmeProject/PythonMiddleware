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

from typing import List, Sequence, TypeVar

T = TypeVar('T')

class Index:
    def __init__(self, index: Sequence[T]):
        self.__index_to_row = {idx: i for i, idx in enumerate(index)}

    def __getitem__(self, item: T) -> int:
        return self.__index_to_row[item]

    def get_rows(self, index: Sequence[T]) -> List[int]:
        return [self.__index_to_row[idx] for idx in index]

    def swap_index(self, idx_i: T, idx_j: T) -> None:
        i, j = self.__index_to_row[idx_i], self.__index_to_row[idx_j]
        self.__index_to_row[idx_i], self.__index_to_row[idx_j] = j, i
