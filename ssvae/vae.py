#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class VAE(object):
    def __init__(
        self,
        hyper_params=None,
        sgd_params=None,
        adagrad_params=None,
        model_params=None
    ):

        if (sgd_params is not None) and (adagrad_params is not None):
            raise ValueError('Error: select only one algorithm')

        self.hyper_params = hyper_params
        self.sgd_params = sgd_params
        self.adagrad_params = adagrad_params
        self.model_params = model_params

        self.rng = np.random.RandomState(hyper_params['rng_seed'])

        self.model_params_ = None
        self.decode_main = None
        self.encode_main = None

    def fit(self, x_datas):
        X = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])
        self.init_model_params(dim_x=x_datas.shape[1])

        lbound, consts = self.get_expr_lbound(X)
        cost = -lbound

        model_params_list = [param for param in self.model_params_.values()]

        if self.sgd_params is not None:
            self.hist = self.sgd_calc(
                x_datas,
                cost,
                consts,
                X,
                model_params_list,
                self.sgd_params,
                self.rng
            )
        else:
            self.hist = self.adagrad_calc(
                x_datas,
                cost,
                consts,
                X,
                model_params_list,
                self.adagrad_params,
                self.rng
            )

    def sgd_calc(self, x_datas, cost, consts, X, model_params, hyper_params, rng):
        n_iters = hyper_params['n_iters']
        learning_rate = hyper_params['learning_rate']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

        gparams = T.grad(
            cost=cost,
            wrt=model_params_list,
            consider_constant=consts
        )
        updates = [(param, param - learning_rate * gparam)
                    for param, gparam in zip(model_params, gparams)]

        train = theano.function(
            inputs=[X],
            outputs=cost,
            updates=updates
        )

        validate = theano.function(
            inputs=[X],
            outputs=cost
        )

        n_samples = x_datas.shape[0]
        cost_history = []

        for i in xrange(n_iters):
            ixs = rng.permutation(n_samples)[:minibatch_size]
            minibatch_cost = train(x_datas[ixs])

            if np.mode(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
        return cost_history


    def adagrad_calc(self, x_datas, cost, consts, X, model_params, hyper_params, rng):
        n_iters = hyper_params['n_iters']
        learning_rate = hyper_params['learning_rate']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

        hs = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in model_params]

        gparams = T.grad(
            cost=cost,
            wrt=model_params,
            consider_constant=consts
        )
        updates = [(param, param - learning_rate / T.sqrt(h) * gparam)
                    for param, gparam, h in zip(model_params, gparams, hs)]
        updates += [(h, h + gparam ** 2) for gparam, h in zip(gparams, hs)]

        train = theano.function(
            inputs=[X],
            outputs=cost,
            updates=updates
        )

        validate = theano.function(
            inputs=[X],
            outputs=cost
        )

        n_samples = x_datas.shape[0]
        cost_history = []

        for i in xrange(n_iters):
            ixs = rng.permutation(n_samples)[:minibatch_size]
            minibatch_cost = train(x_datas[ixs])

            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
        return cost_history


# End of Line.
