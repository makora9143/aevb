#! /usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def shared32(x):
    return theano.shared(np.asarray(x).astype(theano.config.floatX))


class Base_VAE(object):
    def __init__(
        self,
        hyper_params=None,
        optimize_params=None,
        model_params=None,
        model_name=None
    ):
        print model_name, "Model is initialize"
        print '\tlearning_rate:', optimize_params['learning_rate']
        print '\tminibatches:', optimize_params['minibatch_size']
        print '\toptimizer:', hyper_params['optimizer']
        print '\tlearn by:', hyper_params['learning_process']
        print '\tnon linear q:', hyper_params['nonlinear_q']
        print '\tnon linear p:', hyper_params['nonlinear_p']
        print '\ttype px:', hyper_params['type_px']

        self.model_name = model_name

        self.hyper_params = hyper_params
        self.optimize_params = optimize_params
        self.model_params = model_params

        self.rng = np.random.RandomState(hyper_params['rng_seed'])

        self.model_params_ = None
        self.decode_main = None
        self.encode_main = None

    def relu(self, x): return x*(x>0) + 0.01 * x
    def softplus(self, x): return T.log(T.exp(x) + 1)
    def identify(self, x): return x
    def get_name(self): return self.model_name

    def sgd(self, params, gparams, hyper_params):
        learning_rate = hyper_params['learning_rate']
        updates = OrderedDict()

        for param, gparam in zip(params, gparams):
            updates[param] = param + learning_rate * gparam

        return updates

    def adagrad(self, params, gparams, hyper_params):
        updates = OrderedDict()
        learning_rate = hyper_params['learning_rate']

        for param, gparam in zip(params, gparams):
            r = shared32(param.get_value() * 0.)

            r_new = r + T.sqr(gparam)

            param_new = learning_rate / (T.sqrt(r_new) + 1) * gparam

            updates[r] = r_new
            updates[param] = param_new

        return updates

    def rmsProp(self, params, gparams, hyper_params):
        updates = OrderedDict()
        learning_rate = hyper_params['learning_rate']
        beta = 0.9

        for param, gparam in zip(params, gparams):
            r = shared32(param.get_value() * 0.)

            r_new = beta * r + (1 - beta) * T.sqr(gparam)

            param_new = param + learning_rate / (T.sqrt(r_new) + 1) * gparam

            updates[param] = param_new
            updates[r] = r_new
        return updates

    def adaDelta(self, params, gparams, hyper_params):
        learning_rate = hyper_params['learning_rate']
        beta = 0.9

        for param, gparam in zip(params, gparams):
            r = shared32(param.get_value() * 0.)

            v = shared32(param.get_value() * 0.)

            s = shared32(param.get_value() * 0.)

            r_new = beta * r + (1 - beta) * T.sqr(gparam)

            v_new = (T.sqrt(s_new) + 1) / (T.sqrt(v) + 1)

            s_new = beta * s + (1 - beta) * T.sqr(v_new)

            param_new = param + v_new

            updates[s] = s_new
            updates[v] = v_new
            updates[r] = r_new
            updates[param] = param_new
        return updates

    def adam(self, params, gparams, hyper_params):
        updates = OrderedDict()
        decay1 = 0.1
        decay2 = 0.001
        weight_decay = 1000 / 50000.
        learning_rate = hyper_params['learning_rate']

        it = shared32(0.)
        updates[it] = it + 1.

        fix1 = 1. - (1. - decay1) ** (it + 1.)
        fix2 = 1. - (1. - decay2) ** (it + 1.)

        lr_t = learning_rate * T.sqrt(fix2) / fix1

        for param, gparam in zip(params, gparams):
            if weight_decay > 0:
                gparam -= weight_decay * param

            mom1 = shared32(param.get_value(borrow=True) * 0.)
            mom2 = shared32(param.get_value(borrow=True) * 0.)

            mom1_new = mom1 + decay1 * (gparam - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gparam) - mom2)

            effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)

            effstep_new = lr_t * effgrad

            param_new = param + effstep_new

            updates[param] = param_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new

        return updates

# End of Line.
