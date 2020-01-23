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
