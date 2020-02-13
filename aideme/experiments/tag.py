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

import inspect
from typing import Callable, Union, Any


class Tag:
    def __init__(self, obj: Union[type, Callable], **params: Any):
        self.name = obj.__name__
        # self.module = obj.__module__  # TODO: we can also add the module if objects start getting too complex
        self.params = self.check_parameters(obj, params)

    def __repr__(self):
        terms = [self.name]
        for k, v in self.params.items():
            terms.append("{}={}".format(k, v))
        return ' '.join(terms)

    @staticmethod
    def check_parameters(obj, params):
        if inspect.isclass(obj):
            obj = obj.__init__
        elif not inspect.isfunction(obj):
            raise ValueError("Expected 'function' or 'class' object, but got {}".format(obj))

        signature = inspect.signature(obj)

        for param in params:
            if param not in signature.parameters:
                raise TypeError("Unexpected parameter '{}' in object '{}'".format(param, obj.__name__))

        return params

    def to_json(self):
        return {
            'name': self.name,
            'params': {k : self.__check_values(v) for k, v in self.params.items()}
        }

    @staticmethod
    def __check_values(val):
        return val.to_json() if isinstance(val, Tag) else val
