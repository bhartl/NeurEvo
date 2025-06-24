""" A collection of input output (IO) modularity. (c) B. Hartl 2019 """
from .log import Log
from .repr import Repr


def vprint(*args, verbose=True, step=0, **kwargs):
    """ A (non)`verbose` `print` method which can be silenced via the `verbose = False` parameter. """
    if verbose and not step % verbose:
        return print(*args, **kwargs)

    return None
