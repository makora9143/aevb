#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import Layer

def shared32(x):
    return theano.shared(x).astype(theano.config.floatX)

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


    def relu(self, x): return x*(x>0) + 0.01 * x
    def softplus(self, x): return T.log(T.exp(x) + 1)

    def init_model_params(self, dim_x):
        print 'M1 model params initialize'

        dim_z = self.hyper_params['dim_z']
        n_hidden = self.hyper_params['n_hidden'] # [500, 500, 500]
        n_hidden_recognize = n_hidden
        n_hidden_generate = n_hidden[::-1]

        self.type_px = self.hyper_params['type_px']

        activation = {
            'tanh': T.tanh,
            'relu': self.relu,
            'softplus': self.softplus,
            'sigmoid': T.nnet.sigmoid,
            'none': None,
        }

        nonlinear_q = activation[self.hyper_params['nonlinear_q']]
        nonlinear_p = activation[self.hyper_params['nonlinear_p']]
        if self.type_px == 'bernoulli':
            output_f = activation['sigmoid']
        elif self.type_px == 'gaussian':
            output_f= activation['none']

        # Recognize model
        self.recognize_layers = [
            Layer(param_shape=(dim_x, n_hidden[0]), function=nonlinear_q)
        ]
        if len(n_hidden_recognize) > 1:
            self.recognize_layers += [
                Layer(param_shape=shape, function=nonlinear_q)
                for shape in zip(n_hidden_recognize[:-1], n_hidden_recognize[1:])
            ]
        self.recognize_mean_layer = Layer(
            param_shape=(n_hidden_recognize[-1], dim_z),
            function=None
        )
        self.recognize_log_sigma_layer = Layer(
            param_shape=(n_hidden_recognize[-1], dim_z),
            function=None,
            w_zero=True, b_zero=True
        )


        # Generate Model
        self.generate_layers = [
            Layer((dim_z, n_hidden_generate[0]), function=nonlinear_p)
        ]
        if len(n_hidden) > 1:
            self.generate_layers += [
                Layer(param_shape=shape, function=nonlinear_p)
                for shape in zip(n_hidden_generate[:-1], n_hidden_generate[1:])
            ]
        self.generate_mean_layer = Layer(
            param_shape=(n_hidden[-1], dim_x),
            function=output_f
        )
        self.generate_log_sigma_layer = Layer(
            param_shape=(n_hidden[-1], dim_x),
            function=None,
            b_zero=True
        )

        # Add all parameters
        self.model_params_ = (
            [param for layer in self.recognize_layers for param in layer.params] +
            self.recognize_mean_layer.params +
            self.recognize_log_sigma_layer.params +
            [param for layer in self.generate_layers for param in layer.params] +
            self.generate_mean_layer.params
        )

        if self.type_px == 'gaussian':
            self.model_params_ += self.generate_log_sigma_layer.params

    def recognize_model(self, X):
        for i, layer in enumerate(self.recognize_layers):
            if i == 0:
                layer_out = layer.fprop(X)
            else:
                layer_out = layer.fprop(layer_out)

        q_mean = self.recognize_mean_layer.fprop(layer_out)
        q_log_var = self.recognize_log_sigma_layer.fprop(layer_out)

        return {
            'q_mean': q_mean,
            # 'q_log_var': q_log_var,
            'q_log_var': 3 * T.tanh(q_log_var) - 1,
            # 'q_log_var': T.clip(q_log_var, -4., 2.),
        }

    def generate_model(self, Z):
        for i, layer in enumerate(self.generate_layers):
            if i == 0:
                layer_out = layer.fprop(Z)
            else:
                layer_out = layer.fprop(layer_out)

        p_mean = self.generate_mean_layer.fprop(layer_out)
        p_log_var = self.generate_log_sigma_layer.fprop(layer_out)

        return {
            'p_mean': 0.5 * (T.tanh(p_mean) + 1), # 0 <= mu <= 1
            'p_log_var': 3 * T.tanh(p_log_var) - 1, # -4 <= log sigma **2 <= 2
            # 'p_mean': T.clip(p_mean, 0., 1.),
            # 'p_log_var': T.clip(p_log_var, -4., 2.)
            # 'p_mean': p_mean,
            # 'p_log_var': p_log_var
        }

    def encode(self, x):
        if self.encode_main is None:
            X = T.matrix()
            self.encode_main = theano.function(
                inputs=[X],
                outputs=self.recognize_model(X)['q_mean']
            )
        return self.encode_main(x)

    def decode(self, z):
        if self.decode_main is None:
            Z = T.matrix()
            self.decode_main = theano.function(
                inputs=[Z],
                outputs=self.generate_model(Z)['p_mean']
            )
        return self.decode_main(z)

    def get_expr_lbound(self, X):
        n_samples = X.shape[0]

        recognized_zs = self.recognize_model(X)
        q_mean = recognized_zs['q_mean']
        q_log_var = recognized_zs['q_log_var']

        eps = self.rng_noise.normal(size=q_mean.shape).astype(theano.config.floatX)
        # T.exp(0.5 * q_log_var) = std
        # z = mean_z + std * epsilon
        z_tilda = q_mean + T.exp(0.5 * q_log_var) * eps

        generated_x = self.generate_model(z_tilda)
        p_mean = generated_x['p_mean']
        p_log_var = generated_x['p_log_var']

        if self.type_px == 'gaussian':
            log_p_x_given_z = (
                - 0.5 * np.log(2 * np.pi) -
                0.5 * p_log_var -
                0.5 * (X - p_mean) ** 2 / (2 * T.exp(p_log_var))
            )
        elif self.type_px == 'bernoulli':
            log_p_x_given_z = T.sum(X * T.log(p_mean) + (1 - X) * T.log(1 - p_mean), axis=1)

        # logqz = - 0.5 * T.sum(np.log(2 * np.pi) + 1 + q_log_var)
        # logpz = - 0.5 * T.sum(np.log(2 * np.pi) + q_mean ** 2 + T.exp(q_log_var))
        logqz = - 0.5 * T.sum(np.log(2 * np.pi) + 1 + q_log_var, axis=1)
        logpz = - 0.5 * T.sum(np.log(2 * np.pi) + q_mean ** 2 + T.exp(q_log_var), axis=1)

        return (T.sum(log_p_x_given_z) + T.sum(logpz - logqz)) / n_samples,
        # return log_p_x_given_z, logpz, logqz

    def fit(self, x_datas):
        X = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])
        self.init_model_params(dim_x=x_datas.shape[1])

        # logpx, logpz, logqz = self.get_expr_lbound(X)
        # L = -T.sum(logpx + logpz + logqz)
        bound = self.get_expr_lbound(X)
        L = -T.sum(bound)

        print 'start fitting'
        gparams = T.grad(
            cost=L,
            wrt=self.model_params_
        )
        updates = self.adagrad(self.model_params_, gparams, self.adagrad_params)
        self.hist = self.early_stopping(
            X,
            x_datas,
            self.adagrad_params,
            L,
            updates,
            self.rng,
        )

    def adagrad(self, params, gparams, hyper_params):
        learning_rate = hyper_params['learning_rate']

        hs = [theano.shared(np.zeros(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in params]

        updates = [(param, param - learning_rate / (T.sqrt(h) + 1) * gparam)
                    for param, gparam, h in zip(params, gparams, hs)]
        updates += [(h, h + gparam ** 2) for gparam, h in zip(gparams, hs)]
        return updates

    def adam(self, params, gparams, hyper_params):
        learning_rate = hyper_params['learning_rate']

        rs = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in params]
        vs = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in params]
        ts = [theano.shared(np.ones(
                    param.get_value(borrow=True).shape
                ).astype(theano.config.floatX))
            for param in params]

        gnma = 0.999
        beta = 0.9
        weight_decay = 1000 / 50000.

        updates = [
            (param,
             param - learning_rate / (T.sqrt(r / (1 - gnma ** t))) * v / (1 - beta ** t))
             for param, r, v, t  in zip(params, rs, vs, ts)]
        updates += [(r, gnma * r + (1- gnma) * (gparam - weight_decay * param) ** 2) for param, gparam, r in zip(params, gparams, rs)]
        updates += [(v, beta * v + (1- beta) * (gparam - weight_decay * param)) for param, gparam, v in zip(params, gparams, vs)]
        updates += [(t, t + 1) for t in ts]
        return updates



    def optimize(self, X, x_datas, hyper_params, cost, updates, rng):
        n_iters = hyper_params['n_iters']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

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

    def early_stopping(self, X, x_datas, hyper_params, cost, updates, rng):
        n_iters = hyper_params['n_iters']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

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
        best_params = []
        valid_best_error = np.inf
        best_iter = 0
        patient = 0

        for i in xrange(1000000):
            ixs = rng.permutation(n_samples)[:minibatch_size]
            minibatch_cost = train(x_datas[ixs])
            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
                valid_cost = validate(x_datas)
                if valid_cost < valid_best_error:
                    patient = 0
                    best_params = self.model_params_
                    best_iter = i
                else:
                    patient += 1
                if patient > 1000:
                    break
        self.model_params_ = best_params
        return cost_history



# End of Line.
