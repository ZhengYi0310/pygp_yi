"""
Interfaces for parameterized objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import copy

# ABC imports.
#from mwhutils.abc import ABCMeta, abstractmethod
import six
import abc

# exported symbols
__all__ = ['Parameterized', 'printable', 'get_params']

@six.add_metaclass(abc.ABCMeta)
class Parameterized(object):
    """
    Interface for objects that are parameterized by some set of
    hyperparameters.
    """

    @abc.abstractmethod
    def _params(self):
        """
        Define the set of parameters for the model. This should return a list
        of tuples of the form `(name, size, islog)`. If only a 2-tuple is given
        then islog will be assumed to be `True`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_hyper(self):
        """Return a vector of model hyperparameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_hyper(self, hyper):
        """Set the model hyperparameters to the given vector."""
        raise NotImplementedError

    def copy(self, hyper=None):
        """
        Copy the model. If `hyper` is given use this vector to immediately set
        the copied model's hyperparameters.
        """
        model = copy.deepcopy(self)
        if hyper is not None:
            model.set_hyper(hyper)
        return model


def printable(cls):
    """
    Decorator which marks classes as being able to be pretty-printed as a
    function of their hyperparameters. This decorator defines a __repr__ method
    for the given class which uses the class's `get_hyper` and `_params`
    methods to print it.
    """
    def _repr(obj):
        """Represent the object as a function of its hyperparameters."""
        hyper = obj.get_hyper()
        substrings = []
        for key, block, log in get_params(obj):
            val = hyper[block]
            val = val[0] if (len(val) == 1) else val
            val = np.exp(val) if log else val
            substrings += ['%s=%s' % (key, val)]
        return obj.__class__.__name__ + '(' + ', '.join(substrings) + ')'
    cls.__repr__ = _repr
    return cls


# FIXME: the get_params function is kind of a hack in order to allow for
# simpler definitions of the _params() method. This should probably be
# replaced.

def get_params(obj):
    """
    Helper function which translates the values returned by _params() into
    something more meaningful.
    """
    offset = 0
    for param in obj._params():
        key, size, log = param
        block = slice(offset, offset+size)
        offset += size
        yield key, block, log