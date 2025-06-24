import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def locate_type_of(obj):
    cls = obj.__class__
    module = cls.__module__  # inspect.getmodule(self).__name__

    return module, cls
