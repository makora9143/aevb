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

    def fit(self, x_datas, y_labels):
        X = T.matrix()
        Y = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])
        self.init_model_params(dim_x=x_datas.shape[1], dim_y=y_labels.shape[1])

        lbound, consts = self.get_expr_lbound(X, Y)
        cost = -lbound

        model_params_list = [param for param in self.model_params_.values()]

        if self.sgd_params is not None:
            self.hist = self.sgd_calc(
                x_datas,
                y_labels,
                cost,
                consts,
                X,
                Y,
                model_params_list,
                self.sgd_params,
                self.rng
            )
        else:
            self.hist = self.adagrad_calc(
                x_datas,
                y_labels,
                cost,
                consts,
                X,
                Y,
                model_params_list,
                self.adagrad_params,
                self.rng
            )

    def sgd_calc(self, x_datas, y_labels, cost, consts, X, Y, model_params, hyper_params, rng):
        n_iters = hyper_params['n_iters']
        learning_rate = hyper_params['learning_rate']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

        gparams = T.grad(
            cost=cost,
            wrt=model_params,
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

            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs])))
        return cost_history


    def adagrad_calc(self, x_datas, y_labels, cost, consts, X, Y, model_params, hyper_params, rng):
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
            inputs=[X, Y],
            outputs=cost,
            updates=updates
        )

        validate = theano.function(
            inputs=[X, Y],
            outputs=cost
        )

        n_samples = x_datas.shape[0]
        cost_history = []

        for i in xrange(n_iters):
            ixs = rng.permutation(n_samples)[:minibatch_size]
            minibatch_cost = train(x_datas[ixs], y_labels[ixs])

            if np.mod(i, n_mod_history) == 0:
                print '%d epoch error: %f' % (i, minibatch_cost)
                if calc_history == 'minibatch':
                    cost_history.append((i, minibatch_cost))
                else:
                    cost_history.append((i, validate(x_datas[ixs], y_labels[ixs])))
        return cost_history

class M2_VAE(VAE):
    def init_model_params(self, dim_x, dim_y):
        if self.model_params is not None:
            print 'model_params is not None'
            w1 = self.model_params['w1'].astype(theano.config.floatX)
            w2 = self.model_params['w2'].astype(theano.config.floatX)
            w3 = self.model_params['w3'].astype(theano.config.floatX)
            w4 = self.model_params['w4'].astype(theano.config.floatX)
            b1 = self.model_params['b1'].astype(theano.config.floatX)
            b2 = self.model_params['b2'].astype(theano.config.floatX)
            b3 = self.model_params['b3'].astype(theano.config.floatX)

            w5 = self.model_params['w5'].astype(theano.config.floatX)
            w6 = self.model_params['w6'].astype(theano.config.floatX)
            w7 = self.model_params['w7'].astype(theano.config.floatX)
            w8 = self.model_params['w8'].astype(theano.config.floatX)
            b4 = self.model_params['b3'].astype(theano.config.floatX)
            b5 = self.model_params['b3'].astype(theano.config.floatX)
            b6 = self.model_params['b3'].astype(theano.config.floatX)
        else:
            print 'model_params is None'
            dim_z = self.hyper_params['dim_z']
            # dim_y = self.hyper_params['dim_y']
            dim_h_generate = self.hyper_params['dim_h_generate']
            dim_h_recognize = self.hyper_params['dim_h_recognize']

            N = lambda size: self.rng.normal(
                scale=self.hyper_params['scale_init'],
                size=size).astype(theano.config.floatX)
            U = lambda size: self.rng.uniform(
                low=-np.sqrt(6. / np.sum(size)),
                high=np.sqrt(6. / np.sum(size)),
                size=size).astype(theano.config.floatX)

            w1 = U((dim_z, dim_h_generate))
            w2 = U((dim_y, dim_h_generate))
            w3 = U((dim_h_generate, dim_x))
            w4 = U((dim_h_generate, dim_x))
            b1 = N((dim_h_generate,))
            b2 = N((dim_x,))
            b3 = N((dim_x,))

            w5 = U((dim_x, dim_h_recognize))
            w6 = U((dim_y, dim_h_recognize))
            w7 = U((dim_h_recognize, dim_z))
            w8 = U((dim_h_recognize, dim_z))
            b4 = N((dim_h_recognize,))
            b5 = N((dim_z,))
            b6 = N((dim_z,))

        self.model_params_ = {
            'w1': theano.shared(w1),
            'w2': theano.shared(w2),
            'w3': theano.shared(w3),
            'w4': theano.shared(w4),
            'b1': theano.shared(b1),
            'b2': theano.shared(b2),
            'b3': theano.shared(b3),

            'w5': theano.shared(w5),
            'w6': theano.shared(w6),
            'w7': theano.shared(w7),
            'w8': theano.shared(w8),
            'b4': theano.shared(b4),
            'b5': theano.shared(b5),
            'b6': theano.shared(b6),
        }

    def generate_model(self, Z, Y):
        w1 = self.model_params_['w1']
        w2 = self.model_params_['w2']
        w3 = self.model_params_['w3']
        w4 = self.model_params_['w4']
        b1 = self.model_params_['b1']
        b2 = self.model_params_['b2']
        b3 = self.model_params_['b3']

        H = T.tanh(T.dot(Z, w1) + T.dot(Y, w2) + b1)

        return {
            'mu': 0.5 * (T.tanh(T.dot(H, w3) + b2) + 1),
            'log_sigma2': 3 * T.tanh(T.dot(H, w4) + b3) - 1
        }

    def recognize_model(self, X, Y):
        w5 = self.model_params_['w5']
        w6 = self.model_params_['w6']
        w7 = self.model_params_['w7']
        w8 = self.model_params_['w8']
        b4 = self.model_params_['b4']
        b5 = self.model_params_['b5']
        b6 = self.model_params_['b6']

        H = T.tanh(T.dot(X, w5) + T.dot(Y, w6) + b4)

        return {
            'mu': T.dot(H, w7) + b5,
            'log_sigma2': T.dot(H, w8) + b6
        }

    def decode(self, z, y):
        if self.decode_main is None:
            Z = T.matrix()
            Y = T.matrix()
            self.decode_main = theano.function(
                inputs=[Z, Y],
                outputs=self.generate_model(Z, Y)['mu']
            )
        return self.decode_main(z, y)

    def encode(self, x, y):
        if self.encode_main is None:
            X = T.matrix()
            Y = T.matrix()
            self.encode_main = theano.function(
                inputs=[X, Y],
                outputs=self.recognize_model(X, Y)['mu']
            )
        return self.encode_main(x, y)

    def get_expr_lbound(self, X, Y):
        n_mc_sampling = self.hyper_params['n_mc_sampling']
        n_samples = X.shape[0]
        # n_labeled_samples = Y.shape[0]
        # n_unlabeled_samples = n_samples - n_labeled_samples
        dim_z = self.hyper_params['dim_z']

        stats_z = self.recognize_model(X, Y)
        mu_z = stats_z['mu']
        log_sigma2_z = stats_z['log_sigma2']
        sigma2_z = T.exp(log_sigma2_z)

        eps = self.rng_noise.normal(size=(n_mc_sampling, n_samples, dim_z))
        ZS = mu_z + T.sqrt(sigma2_z) * eps

        stats_x = self.generate_model(ZS, Y)
        mu_x = stats_x['mu']
        log_sigma2_x = stats_x['log_sigma2']
        sigma2_x = T.exp(log_sigma2_x)

        log_p_x_given_yz = (
            # - 0.5 * np.log(2 * np.pi) - 0.5 * T.log(sigma2_x) - 0.5 * (X - mu_x) ** 2 / sigma2_x
            - 0.5 * T.log(sigma2_x) - 0.5 * (X - mu_x) ** 2 / sigma2_x
        )

        consts = []

        return (
            0.5 * T.sum(1 + log_sigma2_z - mu_z ** 2 - sigma2_z) / n_samples +
            T.sum(log_p_x_given_yz) / (n_mc_sampling * n_samples)

        ), consts






# End of Line.
