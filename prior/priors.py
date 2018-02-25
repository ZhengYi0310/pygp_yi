"""
Implementations of various prior objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss

# exported symbols
__all__ = ['Uniform', 'Gaussian', 'LogNormal']

class Uniform(object):
    def __init__(self,a , b):
        self.a_ = np.array(a, copy=True, ndmin=1)
        self.b_ = np.array(b, copy=True, ndmin=1)
        self.ndim_ = len(self.a_)

        assert (len(self.a_) == len(self.b_)), "Error: bound sizes don't match!"
        assert(np.all(self.b_ >= self.a_)), "Error: malformed upper/lower bounds"

    def sample(self, size=1, rng=None):
        rng = rstate(rng)
        return self.a_ + (self.b_ - self.a_) * rng.rand(size, self.ndim_)

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        for a, b, t in zip(self.a_, self.b_, theta):
            if (t < a) or (t > b):
                return -np.inf

        return 0.0

class Gaussian(object):
    def __init__(self, mu, var):
        self.mu_ = np.array(mu, copy=True, ndmin=1)
        self.s2_ = np.array(var, copy=True, ndmin=1)
        self.ndim_ = len(self.mu_)
        self.log2pi_ = np.log(2*np.pi)

        if self.s2_.ndim == 1:
            self.std_ = np.sqrt(self.s2_)
        elif self.s2_.ndim == 2:
            self.std_ = np.linalg.cholesky(self.s2_)
        else:
            raise ValueError('Argument `var` can be at most a rank 2 array.')

    def sample(self, size=1, rng=None):
        rng = rstate(rng)
        if self.std_.ndim == 1:
            sample = self.mu_ + self.std_ * rng.randn(size, self.ndim_)
        elif self.s2_.ndim == 2:
            sample = self.mu_ + np.dot(rng.randn(size, self.ndim_), self.std_)
        return sample

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)

        if self.s2_.ndim == 1:
            logpdf = np.sum(np.log(self.s2_))  # log det
            logpdf += np.sum((theta - self.mu_) ** 2 / self.s2_)  # mahalanobis
            logpdf += self.ndim_ * self._log2pi_
            logpdf *= -0.5
        elif self._s2.ndim == 2:
            logpdf = ss.multivariate_normal.logpdf(theta, mean=self.mu_, cov=self.s2_)

        return logpdf

class LogNormal(object):
    def __init__(self, mu=0., sigma=1., min=0.):
        self._mu = np.array(mu, copy=True, ndmin=1)
        self._sigma = np.array(sigma, copy=True, ndmin=1)
        self._min = min
        self.ndim = len(self._mu)

    def sample(self, size=1, rng=None):
        rng = rstate(rng)
        return np.vstack(self._min + rng.lognormal(m, s, size=size)
                         for m, s in zip(self._mu, self._sigma)).T

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        if np.any(theta <= self._min):
            return -np.inf

        logpdf = ss.lognorm.logpdf(theta,
                                   self._sigma,
                                   scale=np.exp(self._mu),
                                   loc=self._min)

        return logpdf.sum()


def rstate(rng=None):
    """
    Return a numpy RandomState object. If an integer value is given then a new
    RandomState will be returned with this seed. If None is given then the
    global numpy state will be returned. If an already instantiated state is
    given this will be passed back.
    """
    if rng is None:
        return np.random.mtrand._rand
    elif isinstance(rng, np.random.RandomState):
        return rng
    elif isinstance(rng, int):
        return np.random.RandomState(rng)
    raise ValueError('unknown seed given to rstate')