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
