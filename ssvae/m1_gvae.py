#! /usr/bin/env python
# -*- coding: utf-8 -*-


import math
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import Layer

class M1_GVAE(object):
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



    def init_model_params(self, dim_x):
        print 'M1 model params initialize'
        dim_z = self.hyper_params['dim_z']
        n_hidden = self.hyper_params['n_hidden'] # [500, 500, 500]
        self.type_px = self.hyper_params['type_px']
        def relu(x): return x*(x>0) + 0.01 * x
        def softplus(x): return T.log(T.exp(x) + 1)
        activation = {'tanh': T.tanh, 'relu': relu, 'softplus': softplus, 'sigmoid': T.nnet.sigmoid, 'none': None}
        nonlinear_q = activation[self.hyper_params['nonlinear_q']]
        nonlinear_p = activation[self.hyper_params['nonlinear_p']]
        if self.type_px == 'bernoulli':
            output_f = activation['sigmoid']
        elif self.type_px == 'gaussian':
            output_f= activation['none']

        # Recognize model
        self.recognize_layers = [Layer((dim_x, n_hidden[0]), function=nonlinear_q)]
        if len(n_hidden) > 1:
            self.recognize_layers += [Layer(shape, function=nonlinear_q) for shape in zip(n_hidden[:-1], n_hidden[1:])]
        self.recognize_mean_layer = Layer((n_hidden[-1], dim_z), function=None)
        self.recognize_log_sigma_layer = Layer((n_hidden[-1], dim_z), function=None, w_zero=True, b_zero=True)


        # Generate Model
        self.generate_layers = [Layer((dim_z, n_hidden[0]), function=nonlinear_p)]
        if len(n_hidden) > 1:
            self.generate_layers += [Layer(shape, function=nonlinear_p) for shape in zip(n_hidden[:-1], n_hidden[1:])]
        self.generate_mean_layer = Layer((n_hidden[-1], dim_x), function=output_f)
        self.generate_log_sigma_layer = Layer((n_hidden[-1], dim_x), function=None, b_zero=True)


        self.model_params_ = (
            [param for layer in self.generate_layers for param in layer.params] +
            self.recognize_mean_layer.params +
            self.recognize_log_sigma_layer.params +
            [param for layer in self.recognize_layers for param in layer.params] +
            self.generate_mean_layer.params
        )

        if self.type_px == 'gaussian':
            self.model_params_ += self.generate_log_sigma_layer.params

    def generate_model(self, Z):

        for i, layer in enumerate(self.generate_layers):
            if i == 0:
                layer_out = layer.fprop(Z)
            else:
                layer_out = layer.fprop(layer_out)

        p_mean = self.generate_mean_layer.fprop(layer_out)
        p_log_var = self.generate_log_sigma_layer.fprop(layer_out)

        return {
            # 'mu': 0.5 * (T.tanh(p_mean) + 1), # 0 <= mu <= 1
            # 'log_sigma': 3 * T.tanh(p_log_var) - 1, # -4 <= log sigma **2 <= 2
            # 'mu': T.clip(p_mean, 0., 1.),
            # 'log_sigma': T.clip(p_log_var, -4., 2.)
            'mu': p_mean,
            'log_sigma': p_log_var
        }

    def recognize_model(self, X):

        for i, layer in enumerate(self.recognize_layers):
            if i == 0:
                layer_out = layer.fprop(X)
            else:
                layer_out = layer.fprop(layer_out)

        q_mean = self.recognize_mean_layer.fprop(layer_out)
        q_log_var = self.recognize_log_sigma_layer.fprop(layer_out)

        return {
            'mu': q_mean,
            # 'log_sigma': 3 * T.tanh(q_log_var) - 1,
            # 'log_sigma': T.clip(q_log_var, -4., 2.)
            'log_sigma': q_log_var
        }

    def decode(self, z):
        if self.decode_main is None:
            Z = T.matrix()
            self.decode_main = theano.function(
                inputs=[Z],
                outputs=self.generate_model(Z)['mu']
            )
        return self.decode_main(z)

    def encode(self, x):
        if self.encode_main is None:
            X = T.matrix()
            self.encode_main = theano.function(
                inputs=[X],
                outputs=self.recognize_model(X)['mu']
            )
        return self.encode_main(x)

    def get_expr_lbound(self, X):
        n_mc_sampling = self.hyper_params['n_mc_sampling']
        n_samples = X.shape[0]
        dim_z = self.hyper_params['dim_z']

        stats_z = self.recognize_model(X)
        q_mean = stats_z['mu']
        q_log_var = stats_z['log_sigma']

        eps = self.rng_noise.normal(size=(n_mc_sampling, n_samples, dim_z))
        z_tilda = q_mean + T.exp(0.5 * q_log_var) * eps

        stats_x = self.generate_model(z_tilda)
        p_mean = stats_x['mu']
        p_log_var = stats_x['log_sigma']

        if self.type_px == 'gaussian':
            log_p_x_given_z = (
                -0.5 * np.log(2 * np.pi) - 0.5 * p_log_var - 0.5 * (X - p_mean) ** 2 / (2 * T.exp(p_log_var))
            )
        elif self.type_px == 'bernoulli':
            log_p_x_given_z = X * T.log(p_mean) + (1 - X) * T.log(1 - p_mean)

        logqz = - 0.5 * T.sum(np.log(2 * np.pi) + 1 + q_log_var)
        logpz = - 0.5 * T.sum(np.log(2 * np.pi) + q_mean ** 2 + T.exp(q_log_var))
        consts = []

        return (T.sum(log_p_x_given_z) / n_mc_sampling + (logpz - logqz)) / n_samples, consts


    def fit(self, x_datas):
        X = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])
        self.init_model_params(dim_x=x_datas.shape[1])

        lbound, consts = self.get_expr_lbound(X)
        cost = -lbound

        print 'start fitting'
        self.hist = self.adam_calc(
            x_datas,
            cost,
            consts,
            X,
            self.model_params_,
            self.adagrad_params,
            self.rng
        )

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
        updates = [(param, param - learning_rate / (T.sqrt(h)) * gparam)
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
            # print minibatch_cost

            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
        return cost_history


    def adam_calc(self, x_datas, cost, consts, X, model_params, hyper_params, rng):
        n_iters = hyper_params['n_iters']
        learning_rate = hyper_params['learning_rate']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

        rs = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in model_params]
        vs = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in model_params]
        ts = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in model_params]

        gnma = 0.999
        beta = 0.9
        weight_decay = 1000 / 50000.

        gparams = T.grad(
            cost=cost,
            wrt=model_params,
            consider_constant=consts
        )


        updates = [(param, param - learning_rate / (T.sqrt(r / (1 - gnma ** t))) * v / (1 - beta ** t))
                    for param, r, v, t  in zip(model_params, rs, vs, ts)]
        updates += [(r, gnma * r + (1- gnma) * (gparam - weight_decay * param) ** 2) for param, gparam, r in zip(model_params, gparams, rs)]
        updates += [(v, beta * v + (1- beta) * (gparam - weight_decay * param)) for param, gparam, v in zip(model_params, gparams, vs)]
        updates += [(t, t + 1) for t in ts]


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
            # print minibatch_cost

            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
        return cost_history
# End of Line.
