#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

rng_default = np.random.RandomState(1234)

def N(size, rng=rng_default, scale=0.01):
    return rng.normal(scale=scale, size=size).astype(theano.config.floatX)

def U(size, rng=rng_default):
    return rng.uniform(
            low=-np.sqrt(6. / np.sum(size)),
            high=np.sqrt(6. / np.sum(size)),
            size=size).astype(theano.config.floatX)

def rand(size, std=1e-2):
    if len(size) == 2:
        return np.random.normal(0, 1, size=size).astype(theano.config.floatX) / np.sqrt(size[0])
    return np.random.normal(0, std, size=size).astype(theano.config.floatX)


class Layer(object):
    def __init__(self, param_shape, rng=rng_default, w_zero=False, b_zero=False, function=T.tanh, nonbias=False):
        self.param_shape = param_shape
        if w_zero:
            self.W = theano.shared(np.zeros(param_shape).astype(theano.config.floatX))
        else:
            # self.W = theano.shared(U(param_shape, rng=rng))
            self.W = theano.shared(rand(param_shape))
        self.params = [self.W]
        self.nonbias = nonbias
        if not self.nonbias:
            if b_zero:
                self.b = theano.shared(np.zeros((param_shape[1],)).astype(theano.config.floatX))
            else:
                # self.b = theano.shared(N((param_shape[1],), rng=rng))
                self.b = theano.shared(rand((param_shape[1],)))
            self.params.append(self.b)

        self.function = function

    def fprop(self, x):
        if self.function is None:
            if not self.nonbias:
                return T.dot(x, self.W) + self.b
            return T.dot(x, self.W)
        else:
            if not self.nonbias:
                return self.function(T.dot(x, self.W) + self.b)
            else:
                return self.function(T.dot(x, self.W))


# End of Line.
