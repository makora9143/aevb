#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T


RNG = np.random.RandomState(1234)


def N(size, rng=RNG, scale=0.01):
    return rng.normal(scale=scale, size=size)


def U(size, rng=RNG):
    return rng.uniform(
            low=-np.sqrt(6. / np.sum(size)),
            high=np.sqrt(6. / np.sum(size)),
            size=size)


def rand(size, rng=RNG, std=1e-2):
    if len(size) == 2:
        return rng.normal(0, 1, size=size) / np.sqrt(size[0])
    return rng.normal(0, std, size=size)


def sharedX(x):
    return theano.shared(np.asarray(x).astype(theano.config.floatX))


class Layer(object):
    def __init__(self, param_shape,
                 rng=RNG, function=T.tanh,
                 w_zero=False, b_zero=False, nonbias=False):
        self.param_shape = param_shape
        self.nonbias = nonbias

        # self.W = theano.shared(U(param_shape, rng=rng))
        self.W = sharedX(rand(param_shape))
        if w_zero:
            self.W = sharedX(np.zeros(param_shape))

        self.params = [self.W]

        if not self.nonbias:
            if b_zero:
                self.b = sharedX(np.zeros((param_shape[1],)))
            else:
                # self.b = theano.shared(N((param_shape[1],), rng=rng))
                self.b = sharedX(rand((param_shape[1],)))
            self.params.append(self.b)

        self.function = function

    def fprop(self, x):
        if not self.nonbias:
            return self.function(T.dot(x, self.W) + self.b)
        else:
            return self.function(T.dot(x, self.W))


# End of Line.
